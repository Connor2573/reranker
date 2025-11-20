from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator


class RerankRequest(BaseModel):
    query: str = Field(..., description="User query or sub-question.")
    documents: List[Any] = Field(..., description="Documents to score. Accepts strings or dicts containing 'text'/'content'.")
    model: Optional[str] = Field(default=None, description="Name of the reranker to use.")
    top_n: Optional[int] = Field(default=None, description="Number of results to return.")
    return_documents: bool = Field(default=True, description="Include original doc payloads in response.")

    @model_validator(mode="after")
    def _validate_top_n(self):  # type: ignore[override]
        if not self.documents:
            raise ValueError("documents must contain at least one entry")
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be > 0")
        return self


class RankedDocument(BaseModel):
    index: int
    score: float
    document: Any | None = None


class RerankResponse(BaseModel):
    model: str
    latency_ms: float
    results: List[RankedDocument]


class HealthResponse(BaseModel):
    status: str
    default_model: str
    device: str


class ModelsResponse(BaseModel):
    models: List[str]
    cached: List[str]
