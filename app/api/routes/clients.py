"""Client management routes."""

from fastapi import APIRouter, HTTPException

from app.api.models import RegisterClientRequest, ClientResponse
from app.clients import manager

router = APIRouter(prefix="/clients", tags=["clients"])


@router.post("/", response_model=ClientResponse)
def register_client(req: RegisterClientRequest) -> ClientResponse:
    """Register a new client."""
    cfg = manager.register(
        client_id=req.client_id,
        display_name=req.display_name,
        coarse_k=req.coarse_k,
        fine_k=req.fine_k,
        max_context_tokens=req.max_context_tokens,
        notes=req.notes,
    )
    stats = manager.client_stats(req.client_id)
    return ClientResponse(
        client_id=cfg.client_id,
        display_name=cfg.display_name,
        created_at=cfg.created_at,
        chunk_count=stats.get("chunk_count", 0),
        notes=cfg.notes,
    )


@router.get("/", response_model=list[ClientResponse])
def list_clients() -> list[ClientResponse]:
    """List all registered clients."""
    results = []
    for cfg in manager.list_clients():
        stats = manager.client_stats(cfg.client_id)
        results.append(
            ClientResponse(
                client_id=cfg.client_id,
                display_name=cfg.display_name,
                created_at=cfg.created_at,
                chunk_count=stats.get("chunk_count", 0),
                notes=cfg.notes,
            )
        )
    return results


@router.get("/{client_id}", response_model=ClientResponse)
def get_client(client_id: str) -> ClientResponse:
    """Get a single client."""
    cfg = manager.get(client_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    stats = manager.client_stats(client_id)
    return ClientResponse(
        client_id=cfg.client_id,
        display_name=cfg.display_name,
        created_at=cfg.created_at,
        chunk_count=stats.get("chunk_count", 0),
        notes=cfg.notes,
    )


@router.delete("/{client_id}")
def delete_client(client_id: str, delete_vectors: bool = False) -> dict:
    """Remove a client from the registry."""
    if not manager.remove(client_id, delete_vectors=delete_vectors):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    return {"deleted": client_id, "vectors_deleted": delete_vectors}
