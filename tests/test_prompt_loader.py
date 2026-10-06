"""Unit tests for prompt_loader.py."""

import json
from pathlib import Path

import pytest

from hinglishlens.prompt_loader import (
    _render_few_shot_assistant_reply,
    build_messages,
    load_prompts_yaml,
)
from hinglishlens.tokenizer import tokenize


def test_few_shot_tokens_match_tokenizer():
    """Assert that for every few-shot example, stored token text equals tokenize(example.text)."""
    prompts = load_prompts_yaml()
    few_shots = prompts.get("few_shot", [])
    assert len(few_shots) > 0, "No few-shot examples found in prompts.yaml"

    for example in few_shots:
        raw_text = example["text"]
        stored_tokens = [t.rsplit("|", 2)[0] for t in example["tokens"]]
        tokenized_tokens = [s.text for s in tokenize(raw_text)]
        assert stored_tokens == tokenized_tokens, (
            f"Token mismatch in few-shot example '{example.get('id')}':\n"
            f"Stored:    {stored_tokens}\n"
            f"Tokenizer: {tokenized_tokens}"
        )


def test_no_test_set_leakage():
    """Assert that no few-shot text appears in data/sentiment_test.csv."""
    test_csv = Path("data/sentiment_test.csv")
    if not test_csv.exists():
        pytest.skip("data/sentiment_test.csv does not exist yet")

    prompts = load_prompts_yaml()
    few_shot_texts = {ex["text"].strip().lower() for ex in prompts.get("few_shot", [])}

    with open(test_csv, "r", encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f if line.strip()]

    for line in lines:
        for fs_text in few_shot_texts:
            assert fs_text not in line, f"Data leakage detected! Few-shot text '{fs_text}' found in test set."


def test_build_messages_zero_shot():
    text = "kya baat hai"
    spans = tokenize(text)
    system, messages = build_messages(text, spans, mode="zero_shot")
    assert len(system) > 0
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert text in messages[0]["content"]


def test_build_messages_few_shot():
    text = "kya baat hai"
    spans = tokenize(text)
    _system, messages = build_messages(text, spans, mode="few_shot")
    prompts = load_prompts_yaml()
    num_few_shots = len(prompts.get("few_shot", []))
    expected_message_count = (num_few_shots * 2) + 1
    assert len(messages) == expected_message_count
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[-1]["role"] == "user"


def test_few_shot_assistant_reply_omits_unchanged_n():
    example = {
        "text": "Aaj ka match",
        "tokens": ["Aaj|HI|Aaj", "ka|HI|ka", "match|EN|game"],
        "sentiment": "positive",
        "confidence": 0.9,
        "emotion": "joy",
        "sarcasm": False,
        "translation_en": "Today's match",
        "rationale": "Praise",
    }
    rendered_str = _render_few_shot_assistant_reply(example)
    parsed = json.loads(rendered_str)
    tokens = parsed["tokens"]

    # "Aaj" -> norm == text -> 'n' omitted
    assert "n" not in tokens[0]
    assert tokens[0]["l"] == "HI"

    # "match" -> norm ("game") != text ("match") -> 'n' included
    assert tokens[2]["n"] == "game"
    assert tokens[2]["l"] == "EN"
