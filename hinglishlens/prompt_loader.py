"""Prompt loader for hinglishlens.

Loads prompts.yaml, caches prompt version, and constructs messages for zero_shot and few_shot modes.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from hinglishlens.tokenizer import Span, tokenize


@lru_cache(maxsize=1)
def load_prompts_yaml(file_path: str = "prompts.yaml") -> dict[str, Any]:
    """Load and cache prompts.yaml content.

    Args:
        file_path: Path to prompts.yaml file.

    Returns:
        Parsed YAML dictionary.

    Raises:
        ValueError: If file cannot be read or parsed.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Prompts file not found: '{file_path}'")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise TypeError("prompts.yaml must be a YAML dictionary")
            return data
    except Exception as e:
        raise ValueError(f"Failed to load prompts file '{file_path}': {e}") from e


def get_prompt_version(file_path: str = "prompts.yaml") -> str:
    """Return the version string from prompts.yaml."""
    prompts = load_prompts_yaml(file_path)
    return str(prompts.get("version", "1.0.0"))


def format_numbered_tokens(tokens: list[Span]) -> str:
    """Format token Spans as '0=token0 1=token1 2=token2'."""
    return " ".join(f"{i}={span.text}" for i, span in enumerate(tokens))


def _render_few_shot_assistant_reply(example: dict[str, Any]) -> str:
    """Render a few_shot example into a compact JSON assistant reply string."""
    raw_tokens: list[str] = example["tokens"]
    token_objs = []
    for idx, raw_t in enumerate(raw_tokens):
        parts = raw_t.rsplit("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid few-shot token format: '{raw_t}'")
        text, lang, norm = parts
        tok_dict: dict[str, Any] = {"i": idx, "l": lang}
        if norm != text:
            tok_dict["n"] = norm
        token_objs.append(tok_dict)

    assistant_data = {
        "tokens": token_objs,
        "sentiment": example["sentiment"],
        "confidence": example["confidence"],
        "emotion": example["emotion"],
        "sarcasm": example["sarcasm"],
        "translation_en": example["translation_en"],
        "rationale": example["rationale"],
    }
    return json.dumps(assistant_data, separators=(",", ":"), ensure_ascii=False)


def build_messages(
    text: str,
    tokens: list[Span],
    mode: str = "few_shot",
    prompts_file: str = "prompts.yaml",
) -> tuple[str, list[dict[str, str]]]:
    """Construct system prompt and chat messages for an input text and tokens.

    Args:
        text: Target Hinglish input string.
        tokens: Token Spans of the target input string.
        mode: Prompt mode ('zero_shot' or 'few_shot').
        prompts_file: Path to prompts.yaml.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    prompts = load_prompts_yaml(prompts_file)
    system_prompt = str(prompts["system"]).strip()
    user_template = str(prompts["user_template"]).strip()

    messages: list[dict[str, str]] = []

    if mode == "few_shot":
        few_shots = prompts.get("few_shot", [])
        for example in few_shots:
            ex_text = example["text"]
            # Re-tokenize example text to get Spans for user message formatting
            ex_spans = tokenize(ex_text)
            ex_numbered = format_numbered_tokens(ex_spans)

            user_msg = user_template.format(text=ex_text, numbered_tokens=ex_numbered)
            assistant_msg = _render_few_shot_assistant_reply(example)

            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    # Append target input user message
    target_numbered = format_numbered_tokens(tokens)
    target_user_msg = user_template.format(text=text, numbered_tokens=target_numbered)
    messages.append({"role": "user", "content": target_user_msg})

    return system_prompt, messages
