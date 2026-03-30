"""Client registry — manages per-client configuration and state.

Persists to data/clients.json so clients survive app restarts.
ChromaDB vectors are already persistent via Docker volume.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from app.db.chroma_client import collection_count, delete_collection

logger = logging.getLogger(__name__)

_REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "clients.json")
_registry: dict[str, "ClientConfig"] = {}


@dataclass
class ClientConfig:
    client_id: str
    display_name: str
    coarse_k: int | None = None
    fine_k: int | None = None
    max_context_tokens: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""


# ── Persistence ───────────────────────────────────────────────────────────────

def _save() -> None:
    """Write the current registry to disk as JSON."""
    try:
        os.makedirs(os.path.dirname(_REGISTRY_FILE), exist_ok=True)
        with open(_REGISTRY_FILE, "w") as f:
            json.dump(
                {cid: asdict(cfg) for cid, cfg in _registry.items()},
                f,
                indent=2,
            )
        logger.debug("Client registry saved (%d clients).", len(_registry))
    except Exception as exc:
        logger.error("Failed to save client registry: %s", exc)


def load_from_disk() -> None:
    """Load the client registry from disk on startup.

    Call this once from app lifespan. Safe to call even if file doesn't exist.
    """
    global _registry
    if not os.path.exists(_REGISTRY_FILE):
        logger.info("No client registry file found — starting fresh.")
        return
    try:
        with open(_REGISTRY_FILE) as f:
            data = json.load(f)
        _registry = {cid: ClientConfig(**cfg) for cid, cfg in data.items()}
        logger.info("Loaded %d client(s) from registry file.", len(_registry))
    except Exception as exc:
        logger.error("Failed to load client registry: %s — starting fresh.", exc)
        _registry = {}


# ── Registry operations ───────────────────────────────────────────────────────

def register(
    client_id: str,
    display_name: str = "",
    coarse_k: int | None = None,
    fine_k: int | None = None,
    max_context_tokens: int | None = None,
    notes: str = "",
) -> ClientConfig:
    """Register a new client (or update existing) and persist to disk."""
    cfg = ClientConfig(
        client_id=client_id,
        display_name=display_name or client_id,
        coarse_k=coarse_k,
        fine_k=fine_k,
        max_context_tokens=max_context_tokens,
        notes=notes,
    )
    _registry[client_id] = cfg
    _save()
    logger.info("Client registered: '%s' (%s)", client_id, display_name)
    return cfg


def get(client_id: str) -> ClientConfig | None:
    return _registry.get(client_id)


def exists(client_id: str) -> bool:
    return client_id in _registry


def list_clients() -> list[ClientConfig]:
    return list(_registry.values())


def remove(client_id: str, delete_vectors: bool = False) -> bool:
    if client_id not in _registry:
        return False
    del _registry[client_id]
    _save()
    if delete_vectors:
        delete_collection(client_id)
    logger.info("Client removed: '%s'", client_id)
    return True


def client_stats(client_id: str) -> dict:
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