"""Document ingestion routes."""

from fastapi import APIRouter, HTTPException

from app.api.models import IngestRequest, IngestResponse
from app.clients import manager
from app.db.chroma_client import delete_collection
from app.ingestion.intake import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Ingest a document for a registered client."""
    if not manager.exists(req.client_id):
        raise HTTPException(
            status_code=404,
            detail=f"Client '{req.client_id}' not found. Register it first via POST /clients/.",
        )

    result = ingest_document(
        text=req.text,
        client_id=req.client_id,
        source=req.source,
        doc_id=req.doc_id,
        extra_metadata=req.extra_metadata,
    )

    return IngestResponse(
        doc_id=result.doc_id,
        client_id=result.client_id,
        total_chunks=result.total_chunks,
        stored_chunks=result.stored_chunks,
        duplicate_chunks=result.duplicate_chunks,
        anomalous_chunks=result.anomalous_chunks,
        errors=result.errors,
    )


@router.delete("/{client_id}")
def clear_corpus(client_id: str) -> dict:
    """Delete all vectors for a client (re-ingest required after this)."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    delete_collection(client_id)
    return {"cleared": client_id}
