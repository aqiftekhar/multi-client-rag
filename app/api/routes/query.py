"""Query route — runs the full multi-agent RAG pipeline."""

from fastapi import APIRouter, HTTPException

from app.api.models import QueryRequest, QueryResponse
from app.agents.orchestrator import Orchestrator
from app.clients import manager

router = APIRouter(prefix="/query", tags=["query"])

# Single shared orchestrator instance (thread-safe — no mutable state per request)
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


@router.post("/", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run a RAG query against a client's corpus."""
    if not manager.exists(req.client_id):
        raise HTTPException(
            status_code=404,
            detail=f"Client '{req.client_id}' not found. Register it first via POST /clients/.",
        )

    result = get_orchestrator().run(query=req.query, client_id=req.client_id)

    return QueryResponse(
        run_id=result["run_id"],
        query=result["query"],
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        retry_count=result["retry_count"],
        errors=result["errors"],
    )
