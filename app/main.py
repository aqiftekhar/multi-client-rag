"""FastAPI application factory."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api.routes import clients, ingest, query, eval as eval_route

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Multi Tenant RAG starting up...")

    # Restore client registry from disk first
    from app.clients.manager import load_from_disk
    load_from_disk()
    from app.clients.manager import load_from_disk
    load_from_disk()

    # Rebuild BM25 indexes from ChromaDB for all registered clients
    # This ensures hybrid search works after app restart without re-ingesting
    from app.rag.bm25_index import rebuild_from_chromadb
    from app.clients.manager import list_clients
    for client_cfg in list_clients():
        rebuild_from_chromadb(client_cfg.client_id)

    from app.evaluation.hallucination_log import load_from_disk as load_hallucination_log
    load_hallucination_log()

    # Warm up embedding model on startup so first query is fast
    try:
        from app.rag.embedder import embed_query
        embed_query("warmup")
        logger.info("Embedding model warmed up.")
    except Exception as exc:
        logger.warning("Embedding warmup failed (non-fatal): %s", exc)
    yield
    logger.info("Multi Tenant RAG shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi Tenant RAG",
        description=(
            "Production-grade multi-agent RAG system with hierarchical retrieval, "
            "embedding drift detection, and CI/CD evaluation pipelines."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routers
    app.include_router(clients.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(eval_route.router)

    # Health check
    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "service": "Multi Tenant-rag"}

    # Serve static UI
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        def root():
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app


app = create_app()
