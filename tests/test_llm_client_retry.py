"""Unit tests for LLMClient retry logic."""


import pytest

from hinglishlens.config import Config
from hinglishlens.llm_client import (
    LLMClient,
    LLMResponse,
    PermanentLLMError,
    TransientLLMError,
)


class DummyLLMClient(LLMClient):
    """Test LLMClient implementation with customizable failure counter."""

    def __init__(self, config: Config, sleep_fn=None):
        super().__init__(config, sleep_fn=sleep_fn)
        self.attempts = 0
        self.transient_fail_count = 0
        self.raise_permanent = False

    def get_model_name(self) -> str:
        return "dummy-model"

    def _raw_generate(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        self.attempts += 1
        if self.raise_permanent:
            raise PermanentLLMError("Fatal auth error")
        if self.attempts <= self.transient_fail_count:
            raise TransientLLMError(f"Rate limited attempt {self.attempts}")
        return LLMResponse(text='{"status": "ok"}', input_tokens=10, output_tokens=10)


def test_transient_error_retry_success():
    cfg = Config()
    cfg.llm.retries.max_attempts = 3
    sleep_calls = []

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    client = DummyLLMClient(cfg, sleep_fn=mock_sleep)
    client.transient_fail_count = 2  # fail 2 times, succeed on 3rd attempt

    resp = client.generate("system", [{"role": "user", "content": "hi"}])
    assert resp.text == '{"status": "ok"}'
    assert client.attempts == 3
    assert len(sleep_calls) == 2


def test_transient_error_exceed_max_attempts():
    cfg = Config()
    cfg.llm.retries.max_attempts = 3
    sleep_calls = []

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    client = DummyLLMClient(cfg, sleep_fn=mock_sleep)
    client.transient_fail_count = 5  # fail 5 times (exceeds max_attempts=3)

    with pytest.raises(TransientLLMError):
        client.generate("system", [{"role": "user", "content": "hi"}])

    assert client.attempts == 3
    assert len(sleep_calls) == 2


def test_permanent_error_no_retry():
    cfg = Config()
    cfg.llm.retries.max_attempts = 3
    sleep_calls = []

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    client = DummyLLMClient(cfg, sleep_fn=mock_sleep)
    client.raise_permanent = True

    with pytest.raises(PermanentLLMError):
        client.generate("system", [{"role": "user", "content": "hi"}])

    # Should fail fast on first attempt without sleeping
    assert client.attempts == 1
    assert len(sleep_calls) == 0
