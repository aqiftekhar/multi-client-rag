"""Client registry — manages per-client configuration and state."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.db.chroma_client import collection_count, delete_collection

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Per-client configuration overrides."""

    client_id: str
    display_name: str
    coarse_k: int | None = None       # override global default
    fine_k: int | None = None         # override global default
    max_context_tokens: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""


# In-memory registry (production: replace with a database)
_registry: dict[str, ClientConfig] = {}


def register(
    client_id: str,
    display_name: str = "",
    coarse_k: int | None = None,
    fine_k: int | None = None,
    max_context_tokens: int | None = None,
    notes: str = "",
) -> ClientConfig:
    """Register a new client (or update existing)."""
    cfg = ClientConfig(
        client_id=client_id,
        display_name=display_name or client_id,
        coarse_k=coarse_k,
        fine_k=fine_k,
        max_context_tokens=max_context_tokens,
        notes=notes,
    )
    _registry[client_id] = cfg
    logger.info("Client registered: '%s' (%s)", client_id, display_name)
    return cfg


def get(client_id: str) -> ClientConfig | None:
    """Return config for *client_id*, or None if not registered."""
    return _registry.get(client_id)


def exists(client_id: str) -> bool:
    return client_id in _registry


def list_clients() -> list[ClientConfig]:
    """Return all registered clients."""
    return list(_registry.values())


def remove(client_id: str, delete_vectors: bool = False) -> bool:
    """Remove a client from the registry.

    If *delete_vectors* is True, also deletes their ChromaDB collection.
    """
    if client_id not in _registry:
        return False
    del _registry[client_id]
    if delete_vectors:
        delete_collection(client_id)
    logger.info("Client removed: '%s'", client_id)
    return True


def client_stats(client_id: str) -> dict:
    """Return basic stats for a client."""
    cfg = get(client_id)
    if not cfg:
        return {}
    return {
        "client_id": client_id,
        "display_name": cfg.display_name,
        "created_at": cfg.created_at,
        "chunk_count": collection_count(client_id),
        "notes": cfg.notes,
    }
