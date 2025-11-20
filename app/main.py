from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Sequence

import torch
from fastapi import FastAPI, HTTPException

from .model_manager import ModelManager
from .schemas import (
    HealthResponse,
    ModelsResponse,
    RankedDocument,
    RerankRequest,
    RerankResponse,
)
from .settings import get_settings


settings = get_settings()
model_manager = ModelManager(settings)

app = FastAPI(
    title="Open WebUI Reranker",
    description="Minimal reranking microservice with GPU acceleration.",
    version="0.1.0",
)


def _chunk(items: Sequence[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalise_documents(documents: Sequence[Any]) -> List[Dict[str, Any]]:
    normalised = []
    for doc in documents:
        raw = doc
        text = None
        if isinstance(doc, str):
            text = doc
        elif isinstance(doc, dict):
            text = doc.get("text") or doc.get("content") or doc.get("body")
            if text is None:
                text = json.dumps(doc, ensure_ascii=False)
        else:
            text = str(doc)

        if not text or not str(text).strip():
            raise HTTPException(status_code=400, detail="All documents must contain text content.")

        normalised.append({"text": text, "raw": raw})
    return normalised


def _score_documents(query: str, docs: List[Dict[str, Any]], batch_size: int, model_bundle):
    scores: List[float] = []
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = model_bundle.device

    for batch in _chunk(docs, batch_size):
        pairs = [(query, item["text"]) for item in batch]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        scores.extend(logits.detach().cpu().tolist())
    return scores


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return HealthResponse(
        status="ok",
        default_model=settings.default_model,
        device=settings.resolved_device(),
    )


@app.get("/models", response_model=ModelsResponse)
def models():
    availability = model_manager.available_models()
    return ModelsResponse(
        models=list(availability.keys()),
        cached=[name for name, loaded in availability.items() if loaded],
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(payload: RerankRequest):
    if len(payload.documents) > settings.max_documents:
        raise HTTPException(
            status_code=400,
            detail=f"Document count {len(payload.documents)} exceeds limit of {settings.max_documents}",
        )

    model_bundle = model_manager.load(payload.model)
    docs = _normalise_documents(payload.documents)
    start = time.perf_counter()
    scores = _score_documents(payload.query, docs, settings.batch_size, model_bundle)
    latency_ms = (time.perf_counter() - start) * 1000

    capped_top_n = min(payload.top_n or len(docs), settings.max_top_n, len(docs))
    ranked_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:capped_top_n]

    results = [
        RankedDocument(
            index=i,
            score=float(scores[i]),
            document=docs[i]["raw"] if payload.return_documents else None,
        )
        for i in ranked_indices
    ]

    return RerankResponse(
        model=model_bundle.name,
        latency_ms=round(latency_ms, 3),
        results=results,
    )

