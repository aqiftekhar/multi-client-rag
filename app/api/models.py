"""FastAPI request and response Pydantic models."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─── Clients ──────────────────────────────────────────────────────────────────

class RegisterClientRequest(BaseModel):
    client_id: str
    display_name: str = ""
    notes: str = ""
    coarse_k: int | None = None
    fine_k: int | None = None
    max_context_tokens: int | None = None


class ClientResponse(BaseModel):
    client_id: str
    display_name: str
    created_at: str
    chunk_count: int
    notes: str


# ─── Ingestion ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    client_id: str
    text: str = Field(..., min_length=10)
    source: str = "manual_upload"
    doc_id: str | None = None
    extra_metadata: dict[str, Any] = {}


class IngestResponse(BaseModel):
    doc_id: str
    client_id: str
    total_chunks: int
    stored_chunks: int
    duplicate_chunks: int
    anomalous_chunks: int
    errors: list[str]


# ─── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    client_id: str
    query: str = Field(..., min_length=3)


class QueryResponse(BaseModel):
    run_id: str
    query: str
    answer: str
    confidence: float
    sources: list[str]
    chunks_used: int
    retry_count: int
    errors: list[str]


# ─── Evaluation ────────────────────────────────────────────────────────────────

class EvalQueryItem(BaseModel):
    query: str
    relevant_chunk_ids: list[str] = []


class RunEvalRequest(BaseModel):
    client_queries: dict[str, list[EvalQueryItem]]
    k: int = 5


class ClientEvalResultResponse(BaseModel):
    client_id: str
    num_queries: int
    metrics: dict[str, float]
    drift_score: float
    signal_summary: dict
    passed: bool
    failure_reasons: list[str]


class EvalReportResponse(BaseModel):
    passed: bool
    client_results: list[ClientEvalResultResponse]
    total_clients: int
    passed_clients: int
    summary: dict


class DriftReportResponse(BaseModel):
    client_id: str
    current_similarity: float
    drift_score: float
    threshold: float
    needs_reindex: bool
    snapshot_count: int
    latest_snapshot_at: str
    history: list[dict]


class SignalSummaryResponse(BaseModel):
    client_id: str
    total_signals: int
    counts: dict[str, int]
    task_completion_rate: float
    retry_rate: float
    failure_rate: float
    recent_signals: list[dict]
