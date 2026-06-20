"""Semantic query cache — returns cached answers for similar queries.

Instead of exact string matching, uses cosine similarity between query
embeddings. A new query hitting similarity >= threshold against a cached
query returns the cached response instantly — skipping retrieval,
LLM call, evaluation, and validation entirely.

Cache is per-client so Client A's answers never leak to Client B.
Cache is persisted to disk and reloaded on startup.

Performance impact on M1 with 4b model:
  Cache miss: 30-60 seconds (full pipeline)
  Cache hit:  < 100ms
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "semantic_cache"
)
_DEFAULT_SIMILARITY_THRESHOLD = 0.95
_MAX_CACHE_SIZE = 500    # per client
_CACHE_TTL_HOURS = 24    # cached entries expire after 24 hours


@dataclass
class CacheEntry:
    """One cached query-response pair."""
    query: str
    embedding: list[float]
    response: dict             # full pipeline response dict
    cached_at: str
    hit_count: int = 0
    client_id: str = ""


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return round(self.cache_hits / self.total_queries, 4)


# Per-client cache stores
_caches: dict[str, list[CacheEntry]] = {}
_stats: dict[str, CacheStats] = {}
_lock = RLock()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _is_expired(entry: CacheEntry) -> bool:
    """Check if a cache entry has exceeded TTL."""
    try:
        from datetime import datetime, timezone, timedelta
        cached = datetime.fromisoformat(entry.cached_at)
        age = datetime.now(timezone.utc) - cached
        return age.total_seconds() > _CACHE_TTL_HOURS * 3600
    except Exception:
        return False


def _get_cache(client_id: str) -> list[CacheEntry]:
    with _lock:
        if client_id not in _caches:
            _caches[client_id] = []
        return _caches[client_id]


def _get_stats(client_id: str) -> CacheStats:
    with _lock:
        if client_id not in _stats:
            _stats[client_id] = CacheStats()
        return _stats[client_id]


# ── Persistence ───────────────────────────────────────────────────────────────

def _cache_file(client_id: str) -> str:
    safe = client_id.replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.json")


def _save_cache(client_id: str) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        entries = _get_cache(client_id)
        with open(_cache_file(client_id), "w") as f:
            json.dump(
                [
                    {
                        "query": e.query,
                        "embedding": e.embedding,
                        "response": e.response,
                        "cached_at": e.cached_at,
                        "hit_count": e.hit_count,
                        "client_id": e.client_id,
                    }
                    for e in entries
                ],
                f,
            )
    except Exception as exc:
        logger.error("Failed to save cache for '%s': %s", client_id, exc)


def load_cache_from_disk(client_ids: list[str]) -> None:
    """Load persisted cache entries on startup."""
    for client_id in client_ids:
        path = _cache_file(client_id)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            entries = []
            for item in data:
                entry = CacheEntry(
                    query=item["query"],
                    embedding=item["embedding"],
                    response=item["response"],
                    cached_at=item["cached_at"],
                    hit_count=item.get("hit_count", 0),
                    client_id=item.get("client_id", client_id),
                )
                if not _is_expired(entry):
                    entries.append(entry)
            with _lock:
                _caches[client_id] = entries
            logger.info(
                "Loaded %d cache entries for client '%s'", len(entries), client_id
            )
        except Exception as exc:
            logger.error("Failed to load cache for '%s': %s", client_id, exc)


# ── Public API ────────────────────────────────────────────────────────────────

def get(
    query_embedding: list[float],
    client_id: str,
    threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> dict | None:
    """Look up a cached response for query_embedding.

    Returns the cached response dict if a similar query was cached,
    otherwise returns None (cache miss).
    """
    stats = _get_stats(client_id)
    stats.total_queries += 1

    cache = _get_cache(client_id)
    if not cache:
        stats.cache_misses += 1
        return None

    # Find most similar cached query
    best_sim = 0.0
    best_entry = None

    for entry in cache:
        if _is_expired(entry):
            continue
        sim = _cosine_similarity(query_embedding, entry.embedding)
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    if best_entry and best_sim >= threshold:
        best_entry.hit_count += 1
        stats.cache_hits += 1
        logger.info(
            "Cache HIT — client='%s' similarity=%.4f hits=%d query='%.40s'",
            client_id, best_sim, best_entry.hit_count, best_entry.query,
        )
        # Return a copy tagged as cache hit
        response = dict(best_entry.response)
        response["cache_hit"] = True
        response["cache_similarity"] = round(best_sim, 4)
        return response

    stats.cache_misses += 1
    logger.debug(
        "Cache MISS — client='%s' best_sim=%.4f < threshold=%.4f",
        client_id, best_sim, threshold,
    )
    return None


def put(
    query: str,
    query_embedding: list[float],
    response: dict,
    client_id: str,
) -> None:
    """Store a query-response pair in the cache.

    Only caches responses with confidence >= 0.6 — low confidence answers
    should not be served from cache as they may be wrong.

    Evicts expired entries and oldest entries when cache is full.
    """
    confidence = response.get("confidence", 0.0)
    if confidence < 0.6:
        logger.debug(
            "Cache PUT skipped — confidence=%.2f < 0.6 (query='%.40s')",
            confidence, query,
        )
        return

    # Also skip NOT FOUND responses — they should not be cached
    answer = response.get("answer", "")
    if answer.startswith("NOT FOUND"):
        logger.debug("Cache PUT skipped — NOT FOUND response not cached")
        return

    with _lock:
        cache = _get_cache(client_id)

        # Remove expired entries first
        before = len(cache)
        cache[:] = [e for e in cache if not _is_expired(e)]
        evicted = before - len(cache)
        if evicted:
            _get_stats(client_id).evictions += evicted

        # Evict oldest entries if at capacity
        while len(cache) >= _MAX_CACHE_SIZE:
            cache.pop(0)
            _get_stats(client_id).evictions += 1

        entry = CacheEntry(
            query=query,
            embedding=query_embedding,
            response=response,
            cached_at=datetime.now(timezone.utc).isoformat(),
            client_id=client_id,
        )
        cache.append(entry)

    _save_cache(client_id)
    logger.info(
        "Cache PUT — client='%s' confidence=%.2f cache_size=%d query='%.40s'",
        client_id, confidence, len(_get_cache(client_id)), query,
    )


def invalidate_client(client_id: str) -> int:
    """Clear all cached entries for a client.

    Call this when new documents are ingested — cached answers may be stale.
    """
    with _lock:
        count = len(_caches.get(client_id, []))
        _caches[client_id] = []
    try:
        path = _cache_file(client_id)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    logger.info("Cache invalidated for client '%s': %d entries removed", client_id, count)
    return count


def get_stats(client_id: str) -> dict:
    """Return cache performance statistics for a client."""
    stats = _get_stats(client_id)
    cache = _get_cache(client_id)
    return {
        "client_id": client_id,
        "total_queries": stats.total_queries,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "hit_rate": stats.hit_rate,
        "evictions": stats.evictions,
        "cached_entries": len([e for e in cache if not _is_expired(e)]),
        "threshold": _DEFAULT_SIMILARITY_THRESHOLD,
        "ttl_hours": _CACHE_TTL_HOURS,
    }


def get_all_entries(client_id: str) -> list[dict]:
    """Return all cache entries for inspection."""
    cache = _get_cache(client_id)
    return [
        {
            "query": e.query[:100],
            "cached_at": e.cached_at,
            "hit_count": e.hit_count,
            "confidence": e.response.get("confidence", 0),
        }
        for e in cache
        if not _is_expired(e)
    ]