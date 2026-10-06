"""Unit tests for cache.py."""

import json
from pathlib import Path

from hinglishlens.cache import ResponseCache


def test_cache_roundtrip(tmp_path: Path):
    db_file = str(tmp_path / "test_cache.sqlite")
    cache = ResponseCache(db_path=db_file, enabled=True)

    key = ResponseCache.make_key(
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_version="1.0.0",
        mode="few_shot",
        temperature=0.0,
        text="kya baat hai",
    )

    raw_json = json.dumps({"sentiment": "positive", "confidence": 0.95})
    cache.set(key, raw_json, input_tokens=100, output_tokens=50)

    cached_item = cache.get(key)
    assert cached_item is not None
    raw, in_tok, out_tok = cached_item
    assert raw == raw_json
    assert in_tok == 100
    assert out_tok == 50


def test_cache_key_sensitivity():
    key1 = ResponseCache.make_key("gemini", "g-2.5", "1.0.0", "few_shot", 0.0, "hello")
    key2 = ResponseCache.make_key("gemini", "g-2.5", "1.0.1", "few_shot", 0.0, "hello")  # diff version
    key3 = ResponseCache.make_key("gemini", "g-2.5", "1.0.0", "zero_shot", 0.0, "hello") # diff mode
    key4 = ResponseCache.make_key("openai", "g-2.5", "1.0.0", "few_shot", 0.0, "hello")  # diff provider

    assert key1 != key2
    assert key1 != key3
    assert key1 != key4


def test_cache_corrupted_entry_ignored(tmp_path: Path):
    db_file = str(tmp_path / "test_cache.sqlite")
    cache = ResponseCache(db_path=db_file, enabled=True)
    key = "test_corrupt_key"

    # Store invalid raw string (not valid JSON object)
    cache.set(key, "NOT_VALID_JSON", input_tokens=10, output_tokens=10)
    assert cache.get(key) is None


def test_cache_disabled(tmp_path: Path):
    db_file = str(tmp_path / "test_cache.sqlite")
    cache = ResponseCache(db_path=db_file, enabled=False)
    key = "any_key"
    cache.set(key, "{}", 10, 10)
    assert cache.get(key) is None
