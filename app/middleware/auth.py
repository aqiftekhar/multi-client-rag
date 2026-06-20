"""API key authentication and per-client rate limiting middleware."""

import logging
import os
import time
from collections import defaultdict
from threading import RLock

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── API Key Auth ──────────────────────────────────────────────────────────────

# Load from environment — set API_KEYS=key1,key2,key3 in .env
# If not set, auth is disabled (development mode)
_raw_keys = os.environ.get("API_KEYS", "")
_VALID_KEYS: set[str] = set(k.strip() for k in _raw_keys.split(",") if k.strip())
_AUTH_ENABLED = bool(_VALID_KEYS)

# Paths that never require auth
_PUBLIC_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc"}


def _is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith("/static")


# ── Rate Limiting ─────────────────────────────────────────────────────────────

_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock = RLock()

# Default limits — can be overridden per route
_RATE_LIMITS = {
    "/query/": {"requests": 10, "window_seconds": 60},     # 10 queries/minute
    "/ingest/": {"requests": 30, "window_seconds": 60},    # 30 ingests/minute
    "/eval/": {"requests": 20, "window_seconds": 60},      # 20 eval calls/minute
}


def _get_rate_limit(path: str) -> dict | None:
    for prefix, limit in _RATE_LIMITS.items():
        if path.startswith(prefix):
            return limit
    return None


def _check_rate_limit(client_key: str, path: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    limit = _get_rate_limit(path)
    if not limit:
        return True

    now = time.time()
    window = limit["window_seconds"]
    max_requests = limit["requests"]

    with _rate_lock:
        # Remove timestamps outside the window
        timestamps = _rate_store[f"{client_key}:{path}"]
        timestamps[:] = [t for t in timestamps if now - t < window]

        if len(timestamps) >= max_requests:
            return False  # Rate limited

        timestamps.append(now)
        return True


# ── Middleware ─────────────────────────────────────────────────────────────────

class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """Combined auth + rate limiting middleware.

    Auth: checks X-API-Key header against configured keys
    Rate limiting: per-key sliding window rate limiting
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for public paths
        if _is_public(path):
            return await call_next(request)

        # ── API Key check ─────────────────────────────────────────────────────
        client_key = "anonymous"
        if _AUTH_ENABLED:
            api_key = request.headers.get("X-API-Key", "")
            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "X-API-Key header required"},
                )
            if api_key not in _VALID_KEYS:
                logger.warning("Invalid API key attempt for path %s", path)
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid API key"},
                )
            client_key = api_key[:8]  # use prefix for rate limiting key
        else:
            logger.debug("Auth disabled — running in development mode")

        # ── Rate limit check ──────────────────────────────────────────────────
        if not _check_rate_limit(client_key, path):
            limit = _get_rate_limit(path)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Max {limit['requests']} requests per {limit['window_seconds']}s for this endpoint."
                },
            )

        return await call_next(request)