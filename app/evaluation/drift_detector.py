"""Embedding drift detection via centroid cosine similarity over time."""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

_MAX_SNAPSHOTS = 10


@dataclass
class DriftSnapshot:
    """A centroid embedding snapshot at a point in time."""

    timestamp: str
    centroid: list[float]
    n_chunks: int


@dataclass
class DriftReport:
    """Result of a drift check."""

    client_id: str
    current_similarity: float          # vs previous snapshot (1.0 = identical)
    drift_score: float                 # 1.0 - similarity (higher = more drift)
    threshold: float
    needs_reindex: bool
    snapshot_count: int
    latest_snapshot_at: str


# Per-client snapshot store
_snapshots: dict[str, deque[DriftSnapshot]] = {}


def _get_snapshots(client_id: str) -> deque[DriftSnapshot]:
    if client_id not in _snapshots:
        _snapshots[client_id] = deque(maxlen=_MAX_SNAPSHOTS)
    return _snapshots[client_id]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def record_snapshot(client_id: str, embeddings: list[list[float]]) -> DriftSnapshot:
    """Compute and store a centroid snapshot from *embeddings*.

    Call this after every re-index or batch ingestion.
    """
    arr = np.array(embeddings)
    centroid = arr.mean(axis=0).tolist()
    now = datetime.now(timezone.utc).isoformat()
    snap = DriftSnapshot(timestamp=now, centroid=centroid, n_chunks=len(embeddings))
    _get_snapshots(client_id).append(snap)
    logger.info(
        "Drift snapshot recorded for client '%s': %d chunks, timestamp=%s",
        client_id,
        len(embeddings),
        now,
    )
    return snap


def check_drift(client_id: str) -> DriftReport:
    """Compare the latest two centroid snapshots for *client_id*.

    If fewer than 2 snapshots exist, reports no drift.
    """
    cfg = get_settings()
    snaps = _get_snapshots(client_id)
    now = datetime.now(timezone.utc).isoformat()

    if len(snaps) < 2:
        return DriftReport(
            client_id=client_id,
            current_similarity=1.0,
            drift_score=0.0,
            threshold=cfg.drift_threshold,
            needs_reindex=False,
            snapshot_count=len(snaps),
            latest_snapshot_at=snaps[-1].timestamp if snaps else now,
        )

    prev, curr = snaps[-2], snaps[-1]
    similarity = _cosine_similarity(prev.centroid, curr.centroid)
    drift_score = 1.0 - similarity
    needs_reindex = drift_score > cfg.drift_threshold

    if needs_reindex:
        logger.warning(
            "Embedding drift detected for client '%s': drift=%.4f > threshold=%.4f",
            client_id,
            drift_score,
            cfg.drift_threshold,
        )

    return DriftReport(
        client_id=client_id,
        current_similarity=similarity,
        drift_score=drift_score,
        threshold=cfg.drift_threshold,
        needs_reindex=needs_reindex,
        snapshot_count=len(snaps),
        latest_snapshot_at=curr.timestamp,
    )


def get_drift_history(client_id: str) -> list[dict]:
    """Return the drift score history as a list of dicts (for API responses)."""
    snaps = list(_get_snapshots(client_id))
    history = []
    for i in range(1, len(snaps)):
        sim = _cosine_similarity(snaps[i - 1].centroid, snaps[i].centroid)
        history.append(
            {
                "timestamp": snaps[i].timestamp,
                "drift_score": round(1.0 - sim, 4),
                "n_chunks": snaps[i].n_chunks,
            }
        )
    return history
