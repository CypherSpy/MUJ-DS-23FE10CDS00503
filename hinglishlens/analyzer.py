"""Core HinglishLens analyzer pipeline with JSON extraction, validation, repair retry, and degradation fallbacks."""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from hinglishlens.cache import ResponseCache
from hinglishlens.cmi import compute_metrics
from hinglishlens.config import Config, load_config
from hinglishlens.llm_client import LLMClient, LLMResponse, PermanentLLMError, get_client
from hinglishlens.prompt_loader import build_messages, get_prompt_version
from hinglishlens.schemas import (
    AnalysisResult,
    LLMOutput,
    ResultMeta,
    Token,
    validate_indices,
)
from hinglishlens.tokenizer import is_language_independent, rebuild, tokenize

logger = logging.getLogger(__name__)


def extract_json(raw: str) -> str:
    """Extract JSON object string from raw text by stripping markdown code fences or finding balanced braces.

    Args:
        raw: Raw output string from LLM.

    Returns:
        JSON string slice.
    """
    text = raw.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line if it's ```json or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Drop last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find first '{' and last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]

    return text


class HinglishAnalyzer:
    """Core Hinglish text normalizer and analyzer."""

    def __init__(
        self,
        config: Config | None = None,
        client: LLMClient | None = None,
        cache: ResponseCache | None = None,
    ):
        self.config = config or load_config()
        self.client = client or get_client(self.config)

        if cache is not None:
            self.cache = cache
        elif self.config.cache.enabled:
            self.cache = ResponseCache(
                db_path=self.config.cache.path,
                enabled=self.config.cache.enabled,
            )
        else:
            self.cache = ResponseCache(enabled=False)

        self.prompt_version = get_prompt_version(self.config.prompt.file)

    def analyze(self, text: str) -> AnalysisResult:
        """Analyze a single Hinglish text string.

        Args:
            text: Raw input string.

        Returns:
            Validated AnalysisResult object.
        """
        start_t = time.perf_counter()

        # Handle empty input
        clean_text = text.strip() if text else ""
        if not clean_text:
            logger.warning("Empty input provided to HinglishAnalyzer.analyze")
            empty_meta = ResultMeta(
                provider=self.config.llm.provider,
                model=self.client.get_model_name(),
                prompt_version=self.prompt_version,
                mode=self.config.prompt.mode,
                latency_ms=round((time.perf_counter() - start_t) * 1000, 2),
                cached=False,
            )
            return AnalysisResult(
                text=text,
                tokens=[],
                normalized_text="",
                sentiment=None,
                confidence=None,
                emotion=None,
                sarcasm=None,
                translation_en=None,
                rationale=None,
                cmi=0.0,
                switch_points=0,
                lang_distribution={"HI": 0, "EN": 0, "NE": 0, "UNIV": 0, "OTHER": 0},
                meta=empty_meta,
                error="empty input",
            )

        # Truncate long inputs
        max_chars = self.config.pipeline.max_input_chars
        if len(clean_text) > max_chars:
            logger.warning(
                "Input text length %d exceeds max_input_chars %d. Truncating.",
                len(clean_text),
                max_chars,
            )
            clean_text = clean_text[:max_chars]

        # Tokenize text
        spans = tokenize(clean_text)
        n_tokens = len(spans)

        # Build prompt messages
        system_prompt, messages = build_messages(
            text=clean_text,
            tokens=spans,
            mode=self.config.prompt.mode,
            prompts_file=self.config.prompt.file,
        )

        # Check Cache
        provider = self.config.llm.provider
        model = self.client.get_model_name()
        mode = self.config.prompt.mode
        temp = self.config.llm.temperature

        cache_key = ResponseCache.make_key(
            provider=provider,
            model=model,
            prompt_version=self.prompt_version,
            mode=mode,
            temperature=temp,
            text=clean_text,
        )

        cached_hit = self.cache.get(cache_key)
        llm_output: LLMOutput | None = None
        cached_flag = False
        repaired_flag = False
        degraded_flag = False
        last_error_msg = ""
        total_in_tokens: int | None = None
        total_out_tokens: int | None = None

        if cached_hit:
            raw_text, total_in_tokens, total_out_tokens = cached_hit
            try:
                json_str = extract_json(raw_text)
                parsed_out = LLMOutput.model_validate_json(json_str)
                validate_indices(parsed_out, n_tokens)
                llm_output = parsed_out
                cached_flag = True
            except Exception as e:  # noqa: BLE001
                logger.warning("Cache hit validation failed: %s. Ignoring cache entry.", e)
                cached_hit = None

        if not cached_flag:
            # Call LLM
            try:
                llm_resp: LLMResponse = self.client.generate(system_prompt, messages)
                raw_text = llm_resp.text
                total_in_tokens = llm_resp.input_tokens
                total_out_tokens = llm_resp.output_tokens

                json_str = extract_json(raw_text)
                try:
                    parsed_out = LLMOutput.model_validate_json(json_str)
                    validate_indices(parsed_out, n_tokens)
                    llm_output = parsed_out
                except Exception as val_err:  # noqa: BLE001
                    last_error_msg = str(val_err)
                    logger.warning("LLM output validation error: %s. Attempting repair...", val_err)

                    # Attempt self-repair retries
                    max_repair = self.config.llm.max_repair_attempts
                    current_messages = list(messages)
                    current_raw = raw_text

                    for attempt in range(max_repair):
                        current_messages.append({"role": "assistant", "content": current_raw})
                        repair_prompt = (
                            f"The previous output was invalid JSON or failed validation schema:\n"
                            f"{last_error_msg}\n"
                            f"Please reply with ONLY a single valid JSON object fixing these errors."
                        )
                        current_messages.append({"role": "user", "content": repair_prompt})

                        try:
                            repair_resp = self.client.generate(system_prompt, current_messages)
                            current_raw = repair_resp.text
                            if repair_resp.input_tokens:
                                total_in_tokens = (total_in_tokens or 0) + repair_resp.input_tokens
                            if repair_resp.output_tokens:
                                total_out_tokens = (total_out_tokens or 0) + repair_resp.output_tokens

                            repair_json = extract_json(current_raw)
                            parsed_repair = LLMOutput.model_validate_json(repair_json)
                            validate_indices(parsed_repair, n_tokens)
                            llm_output = parsed_repair
                            repaired_flag = True
                            raw_text = current_raw
                            logger.info("LLM self-repair succeeded on attempt %d", attempt + 1)
                            break
                        except Exception as r_err:  # noqa: BLE001
                            last_error_msg = str(r_err)
                            logger.warning("LLM self-repair attempt %d failed: %s", attempt + 1, r_err)

                    if llm_output is None:
                        degraded_flag = True

            except PermanentLLMError:
                # Re-raise permanent LLM errors (e.g. missing API keys)
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("LLM generation failed: %s. Falling back to degraded mode.", e)
                last_error_msg = str(e)
                degraded_flag = True

        # Post-process tokens & rebuild text
        final_tokens: list[Token] = []
        normalized_words: list[str] = []

        if not degraded_flag and llm_output is not None:
            # Map LLM outputs to tokens
            tok_map = {t.i: t for t in llm_output.tokens}
            for idx, span in enumerate(spans):
                out_tok = tok_map.get(idx)
                lang_tag = out_tok.l if out_tok else "OTHER"
                norm_val = out_tok.n if (out_tok and out_tok.n) else span.text

                # Deterministic override: force UNIV for language-independent tokens
                if is_language_independent(span.text):
                    lang_tag = "UNIV"
                    norm_val = span.text

                final_tokens.append(
                    Token(
                        text=span.text,
                        lang=lang_tag,
                        normalized=norm_val,
                        start=span.start,
                        end=span.end,
                    )
                )
                normalized_words.append(norm_val)

            # Store in cache on success
            if not cached_flag:
                self.cache.set(
                    key=cache_key,
                    raw=raw_text,
                    input_tokens=total_in_tokens,
                    output_tokens=total_out_tokens,
                )
        else:
            # Degraded Mode Fallback
            logger.warning("Entering degraded mode for text: '%s'", clean_text)
            for span in spans:
                lang_tag = "UNIV" if is_language_independent(span.text) else "OTHER"
                norm_val = span.text
                final_tokens.append(
                    Token(
                        text=span.text,
                        lang=lang_tag,
                        normalized=norm_val,
                        start=span.start,
                        end=span.end,
                    )
                )
                normalized_words.append(norm_val)

        # Rebuild normalized text preserving spacing
        normalized_text = rebuild(clean_text, spans, normalized_words)

        # Compute code-mixing metrics in code
        cmi_val, switch_pts, lang_dist = compute_metrics(final_tokens)

        total_latency = round((time.perf_counter() - start_t) * 1000, 2)
        meta = ResultMeta(
            provider=provider,
            model=model,
            prompt_version=self.prompt_version,
            mode=mode,
            latency_ms=total_latency,
            cached=cached_flag,
            input_tokens=total_in_tokens,
            output_tokens=total_out_tokens,
            repaired=repaired_flag,
            degraded=degraded_flag,
        )

        error_field = f"degraded mode: {last_error_msg}" if degraded_flag else None

        return AnalysisResult(
            text=clean_text,
            tokens=final_tokens,
            normalized_text=normalized_text,
            sentiment=llm_output.sentiment if llm_output else None,
            confidence=llm_output.confidence if llm_output else None,
            emotion=llm_output.emotion if llm_output else None,
            sarcasm=llm_output.sarcasm if llm_output else None,
            translation_en=llm_output.translation_en if llm_output else None,
            rationale=llm_output.rationale if llm_output else None,
            cmi=cmi_val,
            switch_points=switch_pts,
            lang_distribution=lang_dist,
            meta=meta,
            error=error_field,
        )

    def analyze_batch(
        self,
        texts: list[str],
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[AnalysisResult]:
        """Analyze a list of texts concurrently using ThreadPoolExecutor.

        Args:
            texts: List of Hinglish input strings.
            progress_cb: Optional callback function(completed_count, total_count).

        Returns:
            List of AnalysisResult objects matching input order.
        """
        total = len(texts)
        if total == 0:
            return []

        max_workers = self.config.pipeline.max_workers
        results: list[AnalysisResult | None] = [None] * total
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.analyze, text): idx for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results[idx] = res
                except Exception as e:  # noqa: BLE001
                    logger.error("Unhandled error analyzing batch item index %d: %s", idx, e)
                    empty_meta = ResultMeta(
                        provider=self.config.llm.provider,
                        model=self.client.get_model_name(),
                        prompt_version=self.prompt_version,
                        mode=self.config.prompt.mode,
                        latency_ms=0.0,
                        cached=False,
                        degraded=True,
                    )
                    results[idx] = AnalysisResult(
                        text=texts[idx],
                        tokens=[],
                        normalized_text=texts[idx],
                        sentiment=None,
                        confidence=None,
                        emotion=None,
                        sarcasm=None,
                        translation_en=None,
                        rationale=None,
                        cmi=0.0,
                        switch_points=0,
                        lang_distribution={"HI": 0, "EN": 0, "NE": 0, "UNIV": 0, "OTHER": 0},
                        meta=empty_meta,
                        error=f"batch processing error: {e}",
                    )

                completed_count += 1
                if progress_cb:
                    progress_cb(completed_count, total)

        return [r for r in results if r is not None]
