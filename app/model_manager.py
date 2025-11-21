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
    tokenizer: AutoTokenizer | None
    model: AutoModelForSequenceClassification | any
    device: str
    dtype: torch.dtype
    is_jina_v3: bool = False


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

            # Unload all models in single model mode
            if self.settings.single_model_mode:
                self._unload_all()

            device = self.settings.resolved_device()
            dtype = self.settings.resolved_dtype()

            # Detect jina-reranker-v3 which uses custom code
            is_jina_v3 = "jina-reranker-v3" in target_name.lower()

            if is_jina_v3:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    target_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                model.to(device)
                model.eval()
                tokenizer = None
            else:
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
                is_jina_v3=is_jina_v3,
            )
            self._models[target_name] = loaded
            return loaded

    def available_models(self) -> Dict[str, bool]:
        return {name: name in self._models for name in self.settings.allowed_models}

    def _unload_all(self):
        """Unload all models and free GPU memory. Assumes lock is held."""
        for loaded in self._models.values():
            try:
                del loaded.model
                del loaded.tokenizer
            except Exception:  # pragma: no cover
                pass
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

