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
from app.middleware.auth import AuthRateLimitMiddleware

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SkyHi RAG starting up...")

    # 1. Restore client registry
    from app.clients.manager import load_from_disk, list_clients
    load_from_disk()

    # 2. Load persisted signals
    from app.evaluation.signals import load_signals_from_disk
    load_signals_from_disk()

    # 3. Load semantic cache
    from app.rag.semantic_cache import load_cache_from_disk
    load_cache_from_disk([c.client_id for c in list_clients()])

    # 4. Load drift snapshots and re-index logs
    from app.evaluation.drift_detector import (
        load_all_from_disk as load_drift_data,
        start_background_monitor,
    )
    load_drift_data([c.client_id for c in list_clients()])

    # 5. Rebuild BM25 indexes from ChromaDB
    from app.rag.bm25_index import rebuild_from_chromadb
    for client_cfg in list_clients():
        rebuild_from_chromadb(client_cfg.client_id)

    # 6. Start background drift monitor (every 30 minutes)
    start_background_monitor(check_interval_minutes=30)

    # 7. Warm up embedding model
    try:
        from app.rag.embedder import embed_query
        embed_query("warmup")
        logger.info("Embedding model warmed up.")
    except Exception as exc:
        logger.warning("Embedding warmup failed (non-fatal): %s", exc)

    yield

    # Shutdown
    from app.evaluation.drift_detector import stop_background_monitor
    stop_background_monitor()
    logger.info("SkyHi RAG shutting down.")


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

    # Middleware must be registered here in create_app — NOT inside lifespan
    # Starlette raises RuntimeError if add_middleware is called after startup
    app.add_middleware(AuthRateLimitMiddleware)

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