"""Pydantic schemas for LLM outputs, internal tokens, metadata and analysis results."""

from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["HI", "EN", "NE", "UNIV", "OTHER"]
Sentiment = Literal["positive", "negative", "neutral"]
Emotion = Literal["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]


class TokenOut(BaseModel):
    """Token language tag and optional normalized form returned by LLM."""

    i: int
    l: Lang
    n: str | None = None


class LLMOutput(BaseModel):
    """Raw structured output returned by LLM."""

    tokens: list[TokenOut]
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    emotion: Emotion
    sarcasm: bool
    translation_en: str
    rationale: str = Field(max_length=250)


class Token(BaseModel):
    """Final enriched token with span locations and language tag."""

    text: str
    lang: Lang
    normalized: str
    start: int
    end: int


class ResultMeta(BaseModel):
    """Execution metadata for an analysis call."""

    provider: str
    model: str
    prompt_version: str
    mode: str
    latency_ms: float
    cached: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    repaired: bool = False
    degraded: bool = False


class AnalysisResult(BaseModel):
    """Final comprehensive analysis result for a Hinglish text."""

    text: str
    tokens: list[Token]
    normalized_text: str
    sentiment: Sentiment | None = None
    confidence: float | None = None
    emotion: Emotion | None = None
    sarcasm: bool | None = None
    translation_en: str | None = None
    rationale: str | None = None
    cmi: float
    switch_points: int
    lang_distribution: dict[str, int]
    meta: ResultMeta
    error: str | None = None


def validate_indices(output: LLMOutput, n_tokens: int) -> None:
    """Validate that output token indices are exactly {0..n_tokens-1} with no duplicates.

    Args:
        output: Validated LLMOutput object.
        n_tokens: Total number of tokens expected.

    Raises:
        ValueError: If indices are invalid, incomplete, or duplicated.
    """
    indices = [t.i for t in output.tokens]
    if len(indices) != n_tokens:
        raise ValueError(
            f"Token count mismatch: expected {n_tokens} tokens, got {len(indices)}"
        )
    expected_set = set(range(n_tokens))
    actual_set = set(indices)
    if actual_set != expected_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        raise ValueError(
            f"Invalid token indices. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
        )
    if len(actual_set) != len(indices):
        raise ValueError("Duplicate token indices detected in LLM output")
