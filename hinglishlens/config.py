"""Configuration management for HinglishLens.

Loads and validates config.yaml with env var overrides and python-dotenv support.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env if present
load_dotenv()


class AppConfig(BaseModel):
    name: str = "HinglishLens"
    version: str = "1.0.0"


class LLMRetriesConfig(BaseModel):
    max_attempts: int = Field(default=4, ge=1)
    base_delay_s: float = Field(default=1.0, gt=0)
    max_delay_s: float = Field(default=16.0, gt=0)
    jitter: bool = True


class LLMModelsConfig(BaseModel):
    gemini: str = "gemini-2.5-flash"
    openai: str = "gpt-4o-mini"
    anthropic: str = "claude-haiku-4-5-20251001"


class LLMConfig(BaseModel):
    provider: str = Field(default="gemini")
    models: LLMModelsConfig = Field(default_factory=LLMModelsConfig)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1500, ge=1)
    timeout_s: float = Field(default=30.0, gt=0)
    json_mode: bool = True
    retries: LLMRetriesConfig = Field(default_factory=LLMRetriesConfig)
    max_repair_attempts: int = Field(default=1, ge=0)


class PromptConfig(BaseModel):
    file: str = "prompts.yaml"
    mode: str = Field(default="few_shot")


class PipelineConfig(BaseModel):
    max_workers: int = Field(default=4, ge=1)
    max_input_chars: int = Field(default=600, ge=1)


class CacheConfig(BaseModel):
    enabled: bool = True
    path: str = ".cache/llm_cache.sqlite"


class DataConfig(BaseModel):
    train: str = "data/sentiment_train.csv"
    test: str = "data/sentiment_test.csv"
    lid_gold: str = "data/lid_gold.jsonl"
    samples: str = "data/sample_inputs.txt"


class EvaluationConfig(BaseModel):
    output_dir: str = "results"
    seed: int = 42


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/hinglishlens.log"


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str | None = None) -> Config:
    """Load config from YAML file with environment variable overrides.

    Args:
        config_path: Optional path to config file. Defaults to env var
            HINGLISH_CONFIG or 'config.yaml'.

    Returns:
        Validated Config instance.

    Raises:
        ValueError: If config file cannot be read or validated.
    """
    path_str = config_path or os.getenv("HINGLISH_CONFIG") or "config.yaml"
    path = Path(path_str)

    raw_data: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_data = loaded
        except Exception as e:
            raise ValueError(f"Failed to parse config file '{path}': {e}") from e

    # Apply environment variable overrides
    if "LLM_PROVIDER" in os.environ:
        raw_data.setdefault("llm", {})["provider"] = os.environ["LLM_PROVIDER"]

    if "PROMPT_MODE" in os.environ:
        raw_data.setdefault("prompt", {})["mode"] = os.environ["PROMPT_MODE"]

    try:
        config = Config.model_validate(raw_data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration in '{path}':\n{e}") from e

    # Validate provider name
    valid_providers = {"gemini", "openai", "anthropic"}
    if config.llm.provider not in valid_providers:
        raise ValueError(
            f"Invalid llm.provider '{config.llm.provider}'. Must be one of {valid_providers}"
        )

    # Validate prompt mode
    valid_modes = {"zero_shot", "few_shot"}
    if config.prompt.mode not in valid_modes:
        raise ValueError(
            f"Invalid prompt.mode '{config.prompt.mode}'. Must be one of {valid_modes}"
        )

    return config
