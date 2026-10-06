"""Provider-agnostic LLM client with retries, exponential backoff, and exception mapping."""

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
)

from hinglishlens.config import Config

logger = logging.getLogger(__name__)


class HinglishLensError(Exception):
    """Base exception for HinglishLens."""



class TransientLLMError(HinglishLensError):
    """Temporary LLM failure (rate limit, timeout, 5xx) that should be retried."""



class PermanentLLMError(HinglishLensError):
    """Fatal LLM failure (auth, bad request, missing API key) that should fail fast."""



@dataclass
class LLMResponse:
    """Standardized response container for LLM calls."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0


class LLMClient(ABC):
    """Abstract base LLM client."""

    def __init__(self, config: Config, sleep_fn: Callable[[float], None] | None = None):
        self.config = config
        self.sleep_fn = sleep_fn

    @abstractmethod
    def _raw_generate(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        """Provider-specific generation logic.

        Must raise TransientLLMError for retryable errors and
        PermanentLLMError for non-retryable errors.
        """

    def generate(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        """Generate content with tenacity retry and exponential backoff."""
        max_attempts = self.config.llm.retries.max_attempts
        base_delay = self.config.llm.retries.base_delay_s
        max_delay = self.config.llm.retries.max_delay_s
        jitter = self.config.llm.retries.jitter

        if jitter:
            wait_strategy = wait_exponential_jitter(initial=base_delay, max=max_delay)
        else:
            wait_strategy = wait_exponential(initial=base_delay, max=max_delay)

        @retry(
            retry=retry_if_exception_type(TransientLLMError),
            stop=stop_after_attempt(max_attempts),
            wait=wait_strategy,
            sleep=self.sleep_fn or time.sleep,
            reraise=True,
        )
        def _exec_with_retry():
            return self._raw_generate(system, messages)

        start_t = time.perf_counter()
        resp = _exec_with_retry()
        resp.latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        logger.debug(
            "LLM generate call successful. Provider: %s, model: %s, latency: %.2fms, in_tokens: %s, out_tokens: %s",
            self.config.llm.provider,
            self.get_model_name(),
            resp.latency_ms,
            resp.input_tokens,
            resp.output_tokens,
        )
        return resp

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model string used by this client."""


class GeminiClient(LLMClient):
    """Google Gemini API client using google-genai SDK."""

    def __init__(self, config: Config, sleep_fn: Callable[[float], None] | None = None):
        super().__init__(config, sleep_fn)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise PermanentLLMError("GEMINI_API_KEY environment variable is missing")

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except ImportError as e:
            raise PermanentLLMError(f"google-genai package is not installed: {e}") from e
        except Exception as e:
            raise PermanentLLMError(f"Failed to initialize Gemini client: {e}") from e

    def get_model_name(self) -> str:
        return self.config.llm.models.gemini

    def _raw_generate(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        from google.genai import types

        model_name = self.get_model_name()
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

        gen_config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.config.llm.temperature,
            max_output_tokens=self.config.llm.max_output_tokens,
            response_mime_type="application/json" if self.config.llm.json_mode else "text/plain",
        )

        try:
            start = time.perf_counter()
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            text_output = response.text or ""
            in_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            out_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

            return LLMResponse(
                text=text_output,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["429", "rate limit", "quota", "timeout", "500", "503", "unavailable", "connection"]):
                raise TransientLLMError(f"Gemini transient error: {e}") from e
            if any(k in err_msg for k in ["400", "401", "403", "invalid", "unauthorized", "api_key"]):
                raise PermanentLLMError(f"Gemini permanent error: {e}") from e
            # Default to transient for unexpected network issues
            raise TransientLLMError(f"Gemini error: {e}") from e


class OpenAIClient(LLMClient):
    """OpenAI API client using openai SDK."""

    def __init__(self, config: Config, sleep_fn: Callable[[float], None] | None = None):
        super().__init__(config, sleep_fn)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise PermanentLLMError("OPENAI_API_KEY environment variable is missing")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError as e:
            raise PermanentLLMError(f"openai package is not installed: {e}") from e
        except Exception as e:
            raise PermanentLLMError(f"Failed to initialize OpenAI client: {e}") from e

    def get_model_name(self) -> str:
        return self.config.llm.models.openai

    def _raw_generate(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        model_name = self.get_model_name()
        full_messages = [{"role": "system", "content": system}] + messages

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": full_messages,
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_output_tokens,
            "timeout": self.config.llm.timeout_s,
        }
        if self.config.llm.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            start = time.perf_counter()
            response = self.client.chat.completions.create(**kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            choice = response.choices[0]
            text_output = choice.message.content or ""
            usage = response.usage
            in_tokens = usage.prompt_tokens if usage else None
            out_tokens = usage.completion_tokens if usage else None

            return LLMResponse(
                text=text_output,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["429", "rate limit", "quota", "timeout", "500", "502", "503", "connection"]):
                raise TransientLLMError(f"OpenAI transient error: {e}") from e
            if any(k in err_msg for k in ["400", "401", "403", "invalid_api_key", "unauthorized"]):
                raise PermanentLLMError(f"OpenAI permanent error: {e}") from e
            raise TransientLLMError(f"OpenAI error: {e}") from e


class AnthropicClient(LLMClient):
    """Anthropic API client using anthropic SDK."""

    def __init__(self, config: Config, sleep_fn: Callable[[float], None] | None = None):
        super().__init__(config, sleep_fn)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise PermanentLLMError("ANTHROPIC_API_KEY environment variable is missing")

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError as e:
            raise PermanentLLMError(f"anthropic package is not installed: {e}") from e
        except Exception as e:
            raise PermanentLLMError(f"Failed to initialize Anthropic client: {e}") from e

    def get_model_name(self) -> str:
        return self.config.llm.models.anthropic

    def _raw_generate(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        model_name = self.get_model_name()
        try:
            start = time.perf_counter()
            response = self.client.messages.create(
                model=model_name,
                system=system,
                messages=messages,  # type: ignore
                max_tokens=self.config.llm.max_output_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout_s,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            text_output = ""
            if response.content:
                text_output = response.content[0].text  # type: ignore

            in_tokens = getattr(response.usage, "input_tokens", None)
            out_tokens = getattr(response.usage, "output_tokens", None)

            return LLMResponse(
                text=text_output,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["429", "rate limit", "overloaded", "timeout", "500", "529", "connection"]):
                raise TransientLLMError(f"Anthropic transient error: {e}") from e
            if any(k in err_msg for k in ["400", "401", "403", "authentication", "invalid"]):
                raise PermanentLLMError(f"Anthropic permanent error: {e}") from e
            raise TransientLLMError(f"Anthropic error: {e}") from e


def get_client(config: Config, sleep_fn: Callable[[float], None] | None = None) -> LLMClient:
    """Factory function to instantiate the LLMClient for configured provider.

    Args:
        config: HinglishLens Config object.
        sleep_fn: Optional custom sleep function for retry strategy (useful in tests).

    Returns:
        LLMClient subclass instance.

    Raises:
        PermanentLLMError: If provider is unknown or required API key env var is missing.
    """
    provider = config.llm.provider.lower()
    if provider == "gemini":
        return GeminiClient(config, sleep_fn=sleep_fn)
    elif provider == "openai":
        return OpenAIClient(config, sleep_fn=sleep_fn)
    elif provider == "anthropic":
        return AnthropicClient(config, sleep_fn=sleep_fn)
    else:
        raise PermanentLLMError(f"Unsupported LLM provider '{provider}'")
