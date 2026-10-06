"""Unit tests for tokenizer.py."""

import pytest

from hinglishlens.tokenizer import is_language_independent, rebuild, tokenize


def test_tokenize_spans_reconstruct_original():
    text = "Aaj ka match toh ekdum zabardast tha!"
    spans = tokenize(text)
    assert len(spans) > 0
    # Reconstruct text by checking span text matches original slice
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_tokenize_special_tokens():
    text = "Hello @kohli #cricket 100% fine isn't it? 🔥 क्या"
    spans = tokenize(text)
    token_texts = [s.text for s in spans]

    assert "@kohli" in token_texts
    assert "#cricket" in token_texts
    assert "100" in token_texts
    assert "%" in token_texts
    assert "isn't" in token_texts
    assert "🔥" in token_texts
    assert "क्या" in token_texts


def test_rebuild_preserves_spacing():
    text = "Aaj   ka   match ,   very  good !"
    spans = tokenize(text)
    replacements = ["Today", "of", "game", ",", "bahut", "accha", "!"]
    rebuilt = rebuild(text, spans, replacements)
    assert rebuilt == "Today   of   game ,   bahut  accha !"


def test_rebuild_len_mismatch_raises():
    text = "Hello world"
    spans = tokenize(text)
    with pytest.raises(ValueError, match="Length mismatch"):
        rebuild(text, spans, ["Hello"])


def test_is_language_independent():
    assert is_language_independent("123") is True
    assert is_language_independent("!") is True
    assert is_language_independent("🔥") is True
    assert is_language_independent("@user") is True
    assert is_language_independent("#tag") is True
    assert is_language_independent("hello") is False
    assert is_language_independent("kya") is False
    assert is_language_independent("क्या") is False
