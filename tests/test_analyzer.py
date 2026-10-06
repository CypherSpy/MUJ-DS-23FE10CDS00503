"""Unit tests for analyzer.py using FakeLLMClient."""

import json
from pathlib import Path

from hinglishlens.analyzer import HinglishAnalyzer, extract_json
from hinglishlens.config import Config


def test_extract_json():
    raw_fenced = "```json\n{\"sentiment\": \"positive\"}\n```"
    assert extract_json(raw_fenced) == "{\"sentiment\": \"positive\"}"

    raw_text = "Here is the result: {\"a\": 1} thanks"
    assert extract_json(raw_text) == "{\"a\": 1}"


def test_analyzer_empty_input(fake_llm_client, tmp_path: Path):
    cfg = Config()
    cfg.cache.path = str(tmp_path / "cache.sqlite")
    client = fake_llm_client()
    analyzer = HinglishAnalyzer(config=cfg, client=client)

    result = analyzer.analyze("   ")
    assert result.error == "empty input"
    assert client.call_count == 0  # Should not call LLM for empty input


def test_analyzer_success_path(fake_llm_client, tmp_path: Path):
    cfg = Config()
    cfg.cache.path = str(tmp_path / "cache.sqlite")

    valid_json = json.dumps({
        "tokens": [
            {"i": 0, "l": "HI", "n": "kya"},
            {"i": 1, "l": "HI"},
            {"i": 2, "l": "EN"},
            {"i": 3, "l": "UNIV"}
        ],
        "sentiment": "positive",
        "confidence": 0.95,
        "emotion": "joy",
        "sarcasm": False,
        "translation_en": "What a match!",
        "rationale": "Praise"
    })

    client = fake_llm_client([valid_json])
    analyzer = HinglishAnalyzer(config=cfg, client=client)

    res = analyzer.analyze("kyaa baat match !")
    assert res.sentiment == "positive"
    assert res.confidence == 0.95
    assert res.meta.degraded is False
    assert res.meta.repaired is False
    assert len(res.tokens) == 4
    assert res.tokens[0].normalized == "kya"
    assert res.tokens[3].lang == "UNIV"


def test_analyzer_forced_univ_in_code(fake_llm_client, tmp_path: Path):
    cfg = Config()
    cfg.cache.path = str(tmp_path / "cache.sqlite")

    # LLM incorrectly tags '@user' and '#' as 'HI'
    llm_json = json.dumps({
        "tokens": [
            {"i": 0, "l": "HI"},
            {"i": 1, "l": "HI"},
            {"i": 2, "l": "HI"}
        ],
        "sentiment": "neutral",
        "confidence": 0.8,
        "emotion": "neutral",
        "sarcasm": False,
        "translation_en": "hello",
        "rationale": "hello"
    })

    client = fake_llm_client([llm_json])
    analyzer = HinglishAnalyzer(config=cfg, client=client)

    res = analyzer.analyze("hello @user 100")
    # Code must force UNIV for @user and 100
    assert res.tokens[0].lang == "HI"
    assert res.tokens[1].lang == "UNIV"
    assert res.tokens[2].lang == "UNIV"


def test_analyzer_repair_path(fake_llm_client, tmp_path: Path):
    cfg = Config()
    cfg.cache.path = str(tmp_path / "cache.sqlite")
    cfg.llm.max_repair_attempts = 1

    invalid_json = "NOT_JSON_OBJECT"
    valid_json = json.dumps({
        "tokens": [{"i": 0, "l": "HI"}],
        "sentiment": "positive",
        "confidence": 0.9,
        "emotion": "joy",
        "sarcasm": False,
        "translation_en": "Hi",
        "rationale": "Hi"
    })

    client = fake_llm_client([invalid_json, valid_json])
    analyzer = HinglishAnalyzer(config=cfg, client=client)

    res = analyzer.analyze("Haan")
    assert res.meta.repaired is True
    assert res.meta.degraded is False
    assert res.sentiment == "positive"
    assert client.call_count == 2


def test_analyzer_degraded_path(fake_llm_client, tmp_path: Path):
    cfg = Config()
    cfg.cache.path = str(tmp_path / "cache.sqlite")
    cfg.llm.max_repair_attempts = 1

    invalid_json1 = "BAD_JSON_1"
    invalid_json2 = "BAD_JSON_2"

    client = fake_llm_client([invalid_json1, invalid_json2])
    analyzer = HinglishAnalyzer(config=cfg, client=client)

    res = analyzer.analyze("Haan bhai")
    assert res.meta.degraded is True
    assert res.sentiment is None
    assert res.error is not None
    assert "degraded mode" in res.error
    assert len(res.tokens) == 2


def test_analyzer_batch_order_and_isolation(fake_llm_client, tmp_path: Path):
    cfg = Config()
    cfg.cache.path = str(tmp_path / "cache.sqlite")
    cfg.pipeline.max_workers = 2

    valid_json1 = json.dumps({
        "tokens": [{"i": 0, "l": "HI"}],
        "sentiment": "positive", "confidence": 0.9, "emotion": "joy",
        "sarcasm": False, "translation_en": "One", "rationale": "One"
    })
    valid_json2 = json.dumps({
        "tokens": [{"i": 0, "l": "EN"}],
        "sentiment": "negative", "confidence": 0.9, "emotion": "anger",
        "sarcasm": False, "translation_en": "Two", "rationale": "Two"
    })

    client = fake_llm_client([valid_json1, valid_json2])
    analyzer = HinglishAnalyzer(config=cfg, client=client)

    results = analyzer.analyze_batch(["Pehla", "Dusra"])
    assert len(results) == 2
    assert results[0].text == "Pehla"
    assert results[1].text == "Dusra"
    assert results[0].sentiment == "positive"
    assert results[1].sentiment == "negative"
