from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .settings import Settings


@dataclass
class LoadedModel:
    name: str
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: str
    dtype: torch.dtype


class ModelManager:
    """Lazy, thread-safe model cache."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: Dict[str, LoadedModel] = {}
        self._lock = Lock()

        for model_name in self.settings.warm_models:
            try:
                self.load(model_name)
            except Exception as exc:  # pragma: no cover - startup logs
                print(f"[warm] Failed to preload {model_name}: {exc}")

    def load(self, model_name: str | None = None) -> LoadedModel:
        target_name = model_name or self.settings.default_model
        target_name = self.settings.validate_model_name(target_name)

        if target_name in self._models:
            return self._models[target_name]

        with self._lock:
            if target_name in self._models:  # double-check
                return self._models[target_name]

            device = self.settings.resolved_device()
            dtype = self.settings.resolved_dtype()

            tokenizer = AutoTokenizer.from_pretrained(target_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                target_name,
                torch_dtype=dtype,
            )
            model.to(device)
            model.eval()

            loaded = LoadedModel(
                name=target_name,
                tokenizer=tokenizer,
                model=model,
                device=device,
                dtype=dtype,
            )
            self._models[target_name] = loaded
            return loaded

    def available_models(self) -> Dict[str, bool]:
        return {name: name in self._models for name in self.settings.allowed_models}

