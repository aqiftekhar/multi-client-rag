"""ChromaDB client singleton with per-client collection routing."""

import logging
from functools import lru_cache

import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)

_TENANT = "default_tenant"
_DATABASE = "default_database"


def _bootstrap_tenant_and_database(host: str, port: int) -> None:
    """Ensure default_tenant and default_database exist on the ChromaDB server.
    
    ChromaDB HttpClient validates these exist before connecting.
    This bootstraps them idempotently — 409 Conflict means already exists, which is fine.
    """
    base = f"http://{host}:{port}/api/v1"

    with httpx.Client(timeout=10) as http:
        # Create tenant
        r = http.post(f"{base}/tenants", json={"name": _TENANT})
        if r.status_code not in (200, 201, 409):
            logger.warning("Unexpected status creating tenant: %s %s", r.status_code, r.text)

        # Create database inside the tenant
        r = http.post(
            f"{base}/databases",
            json={"name": _DATABASE},
            params={"tenant": _TENANT},
        )
        if r.status_code not in (200, 201, 409):
            logger.warning("Unexpected status creating database: %s %s", r.status_code, r.text)


def _collection_name(client_id: str) -> str:
    safe = client_id.lower().replace("-", "_").replace(" ", "_")
    return f"client_{safe}"


@lru_cache()
def get_chroma_client() -> chromadb.HttpClient:
    cfg = get_settings()

    # MUST run before HttpClient() — that constructor validates the tenant exists
    _bootstrap_tenant_and_database(cfg.chroma_host, cfg.chroma_port)

    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        tenant=_TENANT,
        database=_DATABASE,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.info("ChromaDB connected to %s:%s", cfg.chroma_host, cfg.chroma_port)
    return client


def get_or_create_collection(client_id: str) -> chromadb.Collection:
    chroma = get_chroma_client()
    name = _collection_name(client_id)
    return chroma.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def delete_collection(client_id: str) -> None:
    chroma = get_chroma_client()
    name = _collection_name(client_id)
    try:
        chroma.delete_collection(name)
    except Exception as exc:
        logger.warning("Could not delete collection '%s': %s", name, exc)


def list_collections() -> list[str]:
    return [c.name for c in get_chroma_client().list_collections()]


def collection_count(client_id: str) -> int:
    return get_or_create_collection(client_id).count()