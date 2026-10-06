"""Pytest fixtures and FakeLLMClient for offline testing."""


import pytest

from hinglishlens.config import Config
from hinglishlens.llm_client import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    """Scripted LLM client for offline unit testing without real API calls."""

    def __init__(self, config: Config | None = None, responses: list[str] | None = None):
        cfg = config or Config()
        super().__init__(cfg)
        self.responses = responses or []
        self.call_count = 0
        self.recorded_calls: list[dict] = []

    def get_model_name(self) -> str:
        return "fake-llm-model"

    def _raw_generate(self, system: str, messages: list[dict]) -> LLMResponse:
        self.recorded_calls.append({"system": system, "messages": messages})
        if self.call_count < len(self.responses):
            raw_text = self.responses[self.call_count]
        else:
            # Default fallback valid reply
            raw_text = (
                '{"tokens":[{"i":0,"l":"HI"},{"i":1,"l":"EN"}],'
                '"sentiment":"positive","confidence":0.95,"emotion":"joy",'
                '"sarcasm":false,"translation_en":"Default translation","rationale":"Good"}'
            )
        self.call_count += 1
        return LLMResponse(
            text=raw_text,
            input_tokens=50,
            output_tokens=30,
            latency_ms=10.0,
        )


@pytest.fixture
def fake_llm_client():
    def _factory(responses: list[str] | None = None, config: Config | None = None):
        return FakeLLMClient(config=config, responses=responses)

    return _factory
