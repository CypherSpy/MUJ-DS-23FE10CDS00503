"""Unit tests for schemas.py."""

import pytest
from pydantic import ValidationError

from hinglishlens.schemas import LLMOutput, TokenOut, validate_indices


def test_llm_output_valid():
    out = LLMOutput(
        tokens=[
            TokenOut(i=0, l="HI", n="kya"),
            TokenOut(i=1, l="EN"),
        ],
        sentiment="positive",
        confidence=0.95,
        emotion="joy",
        sarcasm=False,
        translation_en="What",
        rationale="Praise",
    )
    assert out.sentiment == "positive"
    assert out.tokens[0].n == "kya"
    assert out.tokens[1].n is None


def test_llm_output_invalid_confidence():
    with pytest.raises(ValidationError):
        LLMOutput(
            tokens=[TokenOut(i=0, l="HI")],
            sentiment="positive",
            confidence=1.5,  # > 1.0
            emotion="joy",
            sarcasm=False,
            translation_en="Test",
            rationale="Test",
        )


def test_llm_output_invalid_lang():
    with pytest.raises(ValidationError):
        TokenOut(i=0, l="INVALID_LANG")  # type: ignore


def test_validate_indices_success():
    out = LLMOutput(
        tokens=[
            TokenOut(i=0, l="HI"),
            TokenOut(i=1, l="EN"),
            TokenOut(i=2, l="UNIV"),
        ],
        sentiment="neutral",
        confidence=0.8,
        emotion="neutral",
        sarcasm=False,
        translation_en="Test",
        rationale="Test",
    )
    validate_indices(out, 3)  # should not raise


def test_validate_indices_missing_and_duplicate():
    # Duplicate index 0, missing index 1
    out = LLMOutput(
        tokens=[
            TokenOut(i=0, l="HI"),
            TokenOut(i=0, l="EN"),
        ],
        sentiment="neutral",
        confidence=0.8,
        emotion="neutral",
        sarcasm=False,
        translation_en="Test",
        rationale="Test",
    )
    with pytest.raises(ValueError):
        validate_indices(out, 2)
