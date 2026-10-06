"""Regex-based Unicode-aware tokenizer with span preservation."""

import re
from collections.abc import Sequence

from pydantic import BaseModel


class Span(BaseModel):
    """Token span in the original text."""

    text: str
    start: int
    end: int


# Tokenizer regex pattern: @mentions, #hashtags, words (with hyphens/apostrophes/Devanagari), or single non-whitespace punctuation/symbols
TOKEN_PATTERN = re.compile(
    r"[@#]\w+|[\w\u0900-\u097f]+(?:['’\-][\w\u0900-\u097f]+)*|[^\w\s]", re.UNICODE
)


def tokenize(text: str) -> list[Span]:
    """Tokenize input text into Spans with character start and end indices.

    Args:
        text: Raw input text.

    Returns:
        List of Span objects.
    """
    spans: list[Span] = []
    for match in TOKEN_PATTERN.finditer(text):
        spans.append(
            Span(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    return spans


def rebuild(text: str, spans: Sequence[Span], replacements: Sequence[str]) -> str:
    """Rebuild normalized text replacing each span with its replacement while preserving whitespace.

    Args:
        text: Original full text string.
        spans: Sequence of token Spans from tokenize(text).
        replacements: Replacement text corresponding 1-to-1 with spans.

    Returns:
        Reconstructed string with normalized tokens and original spacing.
    """
    if len(spans) != len(replacements):
        raise ValueError(
            f"Length mismatch: {len(spans)} spans vs {len(replacements)} replacements"
        )

    result_parts: list[str] = []
    last_idx = 0

    for span, replacement in zip(spans, replacements):
        # Add intervening whitespace / un-tokenized text
        result_parts.append(text[last_idx : span.start])
        # Add normalized token replacement
        result_parts.append(replacement)
        last_idx = span.end

    # Add remaining trailing text
    result_parts.append(text[last_idx:])
    return "".join(result_parts)


def is_language_independent(token: str) -> bool:
    """Check if a token is inherently language-independent (UNIV).

    Returns True if token has no alphabetic character, or starts with @ or #.

    Args:
        token: Raw or normalized token string.

    Returns:
        bool indicating if token should be forced to UNIV.
    """
    if not token:
        return True
    if token.startswith(("@", "#")):
        return True
    # If no alphabetic characters (e.g. digits, punctuation, emoji)
    return not any(c.isalpha() for c in token)
