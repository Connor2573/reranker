from __future__ import annotations

from functools import lru_cache
from typing import List

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-v2-base",
    "jinaai/jina-reranker-v2-base",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]


class Settings(BaseSettings):
    """Runtime configuration for the reranker service."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_", case_sensitive=False)

    default_model: str = Field(default=_DEFAULT_MODELS[0], description="Model used when the client does not specify one.")
    allowed_models: List[str] = Field(default_factory=lambda: _DEFAULT_MODELS.copy(), description="Whitelisted models that can be loaded.")
    max_documents: int = Field(default=512, description="Upper bound for the number of documents per request.")
    max_top_n: int = Field(default=100, description="Upper bound for top_n to avoid large payloads.")
    max_sequence_length: int = Field(default=512, description="Tokenizer max_length to balance recall and VRAM usage.")
    batch_size: int = Field(default=32, description="How many query-document pairs to score per forward pass.")
    torch_dtype: str | None = Field(default=None, description="Override dtype (float16, bfloat16, float32). Auto-selects if unset.")
    device: str | None = Field(default=None, description="Force computation device (cuda, cpu). Auto-selects if unset.")
    warm_models: List[str] = Field(default_factory=list, description="Models to pre-load on startup.")
    enable_metrics: bool = Field(default=True, description="Expose latency metrics in responses.")

    @field_validator("allowed_models", mode="before")
    @classmethod
    def _split_allowed_models(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("warm_models", mode="before")
    @classmethod
    def _split_warm_models(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_dtype(self):
        if self.torch_dtype:
            return getattr(torch, self.torch_dtype)
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def validate_model_name(self, name: str) -> str:
        if name not in self.allowed_models:
            allowed = ", ".join(self.allowed_models)
            raise ValueError(f"Model '{name}' is not in allowed list ({allowed}).")
        return name


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.default_model not in settings.allowed_models:
        settings.allowed_models.insert(0, settings.default_model)
    return settings
