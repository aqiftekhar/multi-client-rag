"""Production-grade embedding drift detection.

What this implements vs the previous version:

Previous:
  - In-memory snapshots only (lost on restart)
  - Centroid cosine similarity only (misses distribution spread)
  - No re-index logic — only a boolean flag

This version:
  - Snapshots persisted to disk (data/drift_snapshots/{client_id}.json)
  - Three complementary drift signals:
      1. Centroid cosine similarity  — did the mean shift?
      2. Variance drift              — did the spread change?
      3. Jensen-Shannon divergence   — did the distribution shape change?
  - Weighted composite drift score from all three signals
  - Re-index event log (when, why, how many chunks, duration)
  - Automatic re-index trigger with configurable thresholds
  - Background monitoring thread (checks every N minutes)
"""

import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

_MAX_SNAPSHOTS = 50   # keep last 50 — enough for trend analysis
_SNAPSHOT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "drift_snapshots"
)
_REINDEX_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "reindex_log"
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DistributionStats:
    """Statistical summary of an embedding distribution at one point in time."""
    centroid: list[float]          # mean vector
    variance: float                # mean per-dimension variance (scalar)
    std_per_dim: list[float]       # per-dimension standard deviation
    n_chunks: int


@dataclass
class DriftSnapshot:
    """Full snapshot of embedding distribution at one point in time."""
    timestamp: str
    client_id: str
    stats: DistributionStats
    trigger: str = "ingestion"     # "ingestion" | "manual" | "startup"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "trigger": self.trigger,
            "stats": {
                "centroid": self.stats.centroid,
                "variance": self.stats.variance,
                "std_per_dim": self.stats.std_per_dim,
                "n_chunks": self.stats.n_chunks,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DriftSnapshot":
        stats = DistributionStats(
            centroid=data["stats"]["centroid"],
            variance=data["stats"]["variance"],
            std_per_dim=data["stats"]["std_per_dim"],
            n_chunks=data["stats"]["n_chunks"],
        )
        return cls(
            timestamp=data["timestamp"],
            client_id=data["client_id"],
            stats=stats,
            trigger=data.get("trigger", "ingestion"),
        )


@dataclass
class DriftSignals:
    """Individual drift signals computed between two snapshots."""
    centroid_similarity: float     # cosine similarity (1.0 = identical)
    centroid_drift: float          # 1 - centroid_similarity
    variance_drift: float          # abs change in variance (normalised)
    js_divergence: float           # Jensen-Shannon divergence (0 = identical)
    composite_score: float         # weighted combination (0 = no drift, 1 = max)
    n_chunks_before: int
    n_chunks_after: int
    chunks_delta: int              # absolute change in chunk count


@dataclass
class DriftReport:
    """Full drift analysis report for a client."""
    client_id: str
    signals: DriftSignals
    threshold: float
    needs_reindex: bool
    severity: str                  # "none" | "low" | "medium" | "high" | "critical"
    snapshot_count: int
    latest_snapshot_at: str
    recommendation: str
    history: list[dict] = field(default_factory=list)


@dataclass
class ReindexEvent:
    """Record of a re-index operation."""
    event_id: str
    client_id: str
    triggered_at: str
    trigger_reason: str            # "automatic_drift" | "manual" | "scheduled"
    drift_score_before: float
    drift_score_after: float
    chunks_before: int
    chunks_after: int
    duration_seconds: float
    status: str                    # "success" | "failed"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── In-memory stores ──────────────────────────────────────────────────────────

_snapshots: dict[str, deque[DriftSnapshot]] = {}
_reindex_log: dict[str, list[ReindexEvent]] = {}
_lock = threading.RLock()


# ── Persistence ───────────────────────────────────────────────────────────────

def _snapshot_file(client_id: str) -> str:
    safe = client_id.replace("/", "_")
    return os.path.join(_SNAPSHOT_DIR, f"{safe}.json")


def _reindex_file(client_id: str) -> str:
    safe = client_id.replace("/", "_")
    return os.path.join(_REINDEX_LOG_DIR, f"{safe}.json")


def _save_snapshots(client_id: str) -> None:
    try:
        os.makedirs(_SNAPSHOT_DIR, exist_ok=True)
        snaps = list(_snapshots.get(client_id, []))
        with open(_snapshot_file(client_id), "w") as f:
            json.dump([s.to_dict() for s in snaps], f, indent=2)
    except Exception as exc:
        logger.error("Failed to save drift snapshots for '%s': %s", client_id, exc)


def _load_snapshots(client_id: str) -> None:
    path = _snapshot_file(client_id)
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            data = json.load(f)
        dq = deque(maxlen=_MAX_SNAPSHOTS)
        for item in data[-_MAX_SNAPSHOTS:]:
            dq.append(DriftSnapshot.from_dict(item))
        _snapshots[client_id] = dq
        logger.info(
            "Loaded %d drift snapshots for client '%s'", len(dq), client_id
        )
    except Exception as exc:
        logger.error("Failed to load drift snapshots for '%s': %s", client_id, exc)


def _save_reindex_log(client_id: str) -> None:
    try:
        os.makedirs(_REINDEX_LOG_DIR, exist_ok=True)
        events = _reindex_log.get(client_id, [])
        with open(_reindex_file(client_id), "w") as f:
            json.dump([e.to_dict() for e in events], f, indent=2)
    except Exception as exc:
        logger.error("Failed to save re-index log for '%s': %s", client_id, exc)


def _load_reindex_log(client_id: str) -> None:
    path = _reindex_file(client_id)
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            data = json.load(f)
        _reindex_log[client_id] = [ReindexEvent(**e) for e in data]
        logger.info(
            "Loaded %d re-index events for client '%s'",
            len(_reindex_log[client_id]), client_id
        )
    except Exception as exc:
        logger.error("Failed to load re-index log for '%s': %s", client_id, exc)


def load_all_from_disk(client_ids: list[str]) -> None:
    """Load persisted snapshots and re-index logs for all clients on startup."""
    for client_id in client_ids:
        _load_snapshots(client_id)
        _load_reindex_log(client_id)
    logger.info("Drift detector loaded data for %d clients.", len(client_ids))


# ── Distribution stats ────────────────────────────────────────────────────────

def _compute_stats(embeddings: list[list[float]]) -> DistributionStats:
    """Compute distribution statistics from a list of embedding vectors."""
    arr = np.array(embeddings, dtype=np.float32)
    centroid = arr.mean(axis=0)
    # Per-dimension variance then scalar mean
    dim_variance = arr.var(axis=0)
    mean_variance = float(dim_variance.mean())
    std_per_dim = np.sqrt(dim_variance).tolist()
    return DistributionStats(
        centroid=centroid.tolist(),
        variance=mean_variance,
        std_per_dim=std_per_dim,
        n_chunks=len(embeddings),
    )


# ── Drift signal computation ──────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 1.0
    return float(np.dot(va, vb) / (na * nb))


def _variance_drift(prev: DistributionStats, curr: DistributionStats) -> float:
    """Normalised absolute change in embedding variance.

    Returns 0.0 (no change) to 1.0 (extreme change).
    """
    if prev.variance == 0 and curr.variance == 0:
        return 0.0
    denom = max(prev.variance, curr.variance, 1e-8)
    return float(abs(curr.variance - prev.variance) / denom)


def _js_divergence(prev: DistributionStats, curr: DistributionStats) -> float:
    """Approximate Jensen-Shannon divergence between two Gaussian distributions.

    Uses per-dimension std to approximate the distributions.
    Returns 0.0 (identical) to 1.0 (completely different).
    """
    p_std = np.array(prev.std_per_dim, dtype=np.float64) + 1e-8
    q_std = np.array(curr.std_per_dim, dtype=np.float64) + 1e-8
    p_mean = np.array(prev.centroid, dtype=np.float64)
    q_mean = np.array(curr.centroid, dtype=np.float64)

    # KL(P||M) + KL(Q||M) where M is the mixture
    m_mean = (p_mean + q_mean) / 2
    m_std = np.sqrt((p_std**2 + q_std**2) / 2)

    def kl_gaussian(mu1, s1, mu2, s2):
        return np.mean(
            np.log(s2 / s1) + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2) - 0.5
        )

    try:
        kl_pm = kl_gaussian(p_mean, p_std, m_mean, m_std)
        kl_qm = kl_gaussian(q_mean, q_std, m_mean, m_std)
        jsd = float((kl_pm + kl_qm) / 2)
        # Clamp to [0, 1]
        return max(0.0, min(1.0, jsd / 10.0))
    except Exception:
        return 0.0


def _compute_signals(prev: DriftSnapshot, curr: DriftSnapshot) -> DriftSignals:
    """Compute all drift signals between two snapshots."""
    centroid_sim = _cosine_similarity(prev.stats.centroid, curr.stats.centroid)
    centroid_drift = 1.0 - centroid_sim
    var_drift = _variance_drift(prev.stats, curr.stats)
    jsd = _js_divergence(prev.stats, curr.stats)

    # Weighted composite score
    # Centroid drift is most reliable, JSD catches distribution shape changes,
    # variance drift catches spread changes
    composite = (
        centroid_drift * 0.50
        + jsd * 0.35
        + var_drift * 0.15
    )
    composite = max(0.0, min(1.0, composite))

    chunks_delta = curr.stats.n_chunks - prev.stats.n_chunks

    return DriftSignals(
        centroid_similarity=round(centroid_sim, 4),
        centroid_drift=round(centroid_drift, 4),
        variance_drift=round(var_drift, 4),
        js_divergence=round(jsd, 4),
        composite_score=round(composite, 4),
        n_chunks_before=prev.stats.n_chunks,
        n_chunks_after=curr.stats.n_chunks,
        chunks_delta=chunks_delta,
    )


def _severity(composite: float, threshold: float) -> str:
    if composite < threshold * 0.5:
        return "none"
    elif composite < threshold:
        return "low"
    elif composite < threshold * 1.5:
        return "medium"
    elif composite < threshold * 2.0:
        return "high"
    return "critical"


def _recommendation(signals: DriftSignals, threshold: float) -> str:
    if signals.composite_score < threshold * 0.5:
        return "Embeddings stable. No action needed."
    if signals.composite_score < threshold:
        return (
            "Minor drift detected. Monitor over next few ingestions. "
            "No immediate re-index needed."
        )
    if signals.chunks_delta > 0:
        return (
            f"Significant drift after adding {signals.chunks_delta} chunks. "
            "Re-indexing recommended to restore retrieval quality."
        )
    if signals.variance_drift > 0.3:
        return (
            "Distribution spread has changed significantly. "
            "Documents may have shifted in topic. Re-indexing recommended."
        )
    return (
        "Embedding distribution has shifted beyond threshold. "
        "Re-indexing recommended to restore retrieval quality."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def _get_snapshots(client_id: str) -> deque[DriftSnapshot]:
    with _lock:
        if client_id not in _snapshots:
            _snapshots[client_id] = deque(maxlen=_MAX_SNAPSHOTS)
        return _snapshots[client_id]


def record_snapshot(
    client_id: str,
    embeddings: list[list[float]],
    trigger: str = "ingestion",
) -> DriftSnapshot:
    """Compute distribution stats and store a snapshot.

    Called from intake.py after every ingestion.
    Persists to disk immediately.
    """
    if not embeddings:
        logger.warning("record_snapshot called with empty embeddings for '%s'", client_id)
        return None

    stats = _compute_stats(embeddings)
    now = datetime.now(timezone.utc).isoformat()
    snap = DriftSnapshot(
        timestamp=now,
        client_id=client_id,
        stats=stats,
        trigger=trigger,
    )

    with _lock:
        _get_snapshots(client_id).append(snap)
        _save_snapshots(client_id)

    logger.info(
        "Drift snapshot recorded — client='%s' chunks=%d variance=%.4f trigger=%s",
        client_id, stats.n_chunks, stats.variance, trigger,
    )
    return snap


def check_drift(client_id: str) -> DriftReport:
    """Compute full drift report for a client.

    Compares the two most recent snapshots using three signals:
    centroid cosine distance, variance change, and JS divergence.
    """
    cfg = get_settings()
    snaps = list(_get_snapshots(client_id))
    now = datetime.now(timezone.utc).isoformat()

    empty_signals = DriftSignals(
        centroid_similarity=1.0, centroid_drift=0.0,
        variance_drift=0.0, js_divergence=0.0,
        composite_score=0.0,
        n_chunks_before=0, n_chunks_after=snaps[-1].stats.n_chunks if snaps else 0,
        chunks_delta=0,
    )

    if len(snaps) < 2:
        return DriftReport(
            client_id=client_id,
            signals=empty_signals,
            threshold=cfg.drift_threshold,
            needs_reindex=False,
            severity="none",
            snapshot_count=len(snaps),
            latest_snapshot_at=snaps[-1].timestamp if snaps else now,
            recommendation="Not enough snapshots for drift analysis. Ingest more documents.",
            history=[],
        )

    prev, curr = snaps[-2], snaps[-1]
    signals = _compute_signals(prev, curr)
    needs_reindex = signals.composite_score > cfg.drift_threshold
    sev = _severity(signals.composite_score, cfg.drift_threshold)
    rec = _recommendation(signals, cfg.drift_threshold)

    if needs_reindex:
        logger.warning(
            "Drift threshold exceeded — client='%s' composite=%.4f > threshold=%.4f severity=%s",
            client_id, signals.composite_score, cfg.drift_threshold, sev,
        )

    history = _build_history(snaps)

    return DriftReport(
        client_id=client_id,
        signals=signals,
        threshold=cfg.drift_threshold,
        needs_reindex=needs_reindex,
        severity=sev,
        snapshot_count=len(snaps),
        latest_snapshot_at=curr.timestamp,
        recommendation=rec,
        history=history,
    )


def _build_history(snaps: list[DriftSnapshot]) -> list[dict]:
    """Build drift score history for charting."""
    history = []
    for i in range(1, len(snaps)):
        signals = _compute_signals(snaps[i - 1], snaps[i])
        history.append({
            "timestamp": snaps[i].timestamp,
            "composite_score": signals.composite_score,
            "centroid_drift": signals.centroid_drift,
            "variance_drift": signals.variance_drift,
            "js_divergence": signals.js_divergence,
            "n_chunks": snaps[i].stats.n_chunks,
            "chunks_delta": signals.chunks_delta,
            "trigger": snaps[i].trigger,
        })
    return history


def get_drift_history(client_id: str) -> list[dict]:
    """Return full drift history for a client (for API and UI)."""
    snaps = list(_get_snapshots(client_id))
    return _build_history(snaps)


# ── Re-index management ───────────────────────────────────────────────────────

def trigger_reindex(
    client_id: str,
    trigger_reason: str = "automatic_drift",
) -> ReindexEvent:
    """Execute re-indexing for a client and log the event.

    Re-indexing = pull all current embeddings from ChromaDB,
    recompute them fresh, upsert back, take new snapshot.

    Returns the ReindexEvent record.
    """
    import uuid as _uuid
    event_id = str(_uuid.uuid4())[:12]
    started_at = time.time()
    now = datetime.now(timezone.utc).isoformat()

    # Get current state before re-index
    drift_before = check_drift(client_id)
    chunks_before = drift_before.signals.n_chunks_after

    logger.info(
        "Re-index started — client='%s' event_id=%s reason=%s chunks=%d",
        client_id, event_id, trigger_reason, chunks_before,
    )

    status = "success"
    error = ""
    chunks_after = 0
    drift_score_after = 0.0

    try:
        from app.db.chroma_client import get_or_create_collection
        from app.rag.embedder import embed_texts

        collection = get_or_create_collection(client_id)
        total = collection.count()
        if total == 0:
            logger.warning("Re-index: no chunks found for client '%s'", client_id)
            status = "success"
            chunks_after = 0
        else:
            # Fetch all chunks in batches
            batch_size = 500
            all_ids, all_raw_texts, all_metadatas, all_docs = [], [], [], []

            result = collection.get(
                include=["documents", "metadatas"],
                limit=total,
            )
            all_ids = result["ids"]
            all_docs = result["documents"]
            all_metadatas = result["metadatas"]

            # Use raw_text from metadata for re-embedding
            for meta, doc in zip(all_metadatas, all_docs):
                raw = meta.get("raw_text") or doc or ""
                all_raw_texts.append(raw)

            # Re-embed in batches
            new_embeddings = []
            for i in range(0, len(all_raw_texts), batch_size):
                batch = all_raw_texts[i:i + batch_size]
                batch_embs = embed_texts(batch)
                new_embeddings.extend(batch_embs)
                logger.debug(
                    "Re-index batch %d/%d for client '%s'",
                    i // batch_size + 1,
                    math.ceil(len(all_raw_texts) / batch_size),
                    client_id,
                )

            # Upsert with fresh embeddings
            collection.upsert(
                ids=all_ids,
                documents=all_docs,
                metadatas=all_metadatas,
                embeddings=new_embeddings,
            )
            chunks_after = len(all_ids)

            # Take new snapshot after re-index
            record_snapshot(client_id, new_embeddings, trigger="reindex")

            # Check drift score after
            report_after = check_drift(client_id)
            drift_score_after = report_after.signals.composite_score

            logger.info(
                "Re-index complete — client='%s' chunks=%d drift_before=%.4f drift_after=%.4f",
                client_id, chunks_after,
                drift_before.signals.composite_score,
                drift_score_after,
            )

            # Also rebuild BM25 index
            try:
                from app.rag.bm25_index import rebuild_from_chromadb
                rebuild_from_chromadb(client_id)
            except Exception as bm25_exc:
                logger.warning("BM25 rebuild failed during re-index: %s", bm25_exc)

    except Exception as exc:
        status = "failed"
        error = str(exc)
        logger.error("Re-index failed for client '%s': %s", client_id, exc)

    duration = round(time.time() - started_at, 2)

    event = ReindexEvent(
        event_id=event_id,
        client_id=client_id,
        triggered_at=now,
        trigger_reason=trigger_reason,
        drift_score_before=round(drift_before.signals.composite_score, 4),
        drift_score_after=round(drift_score_after, 4),
        chunks_before=chunks_before,
        chunks_after=chunks_after,
        duration_seconds=duration,
        status=status,
        error=error,
    )

    with _lock:
        if client_id not in _reindex_log:
            _reindex_log[client_id] = []
        _reindex_log[client_id].append(event)
        _save_reindex_log(client_id)

    return event


def get_reindex_log(client_id: str) -> list[ReindexEvent]:
    """Return all re-index events for a client."""
    return list(_reindex_log.get(client_id, []))


# ── Background monitoring ─────────────────────────────────────────────────────

_monitor_thread: threading.Thread | None = None
_monitor_stop = threading.Event()


def start_background_monitor(
    check_interval_minutes: int = 30,
) -> None:
    """Start background thread that periodically checks drift for all clients.

    Automatically triggers re-index when composite score exceeds threshold.
    """
    global _monitor_thread
    if _monitor_thread and _monitor_thread.is_alive():
        return

    _monitor_stop.clear()

    def _monitor_loop():
        logger.info(
            "Drift monitor started — checking every %d minutes", check_interval_minutes
        )
        while not _monitor_stop.wait(timeout=check_interval_minutes * 60):
            try:
                from app.clients.manager import list_clients
                clients = list_clients()
                for client_cfg in clients:
                    cid = client_cfg.client_id
                    snaps = list(_get_snapshots(cid))
                    if len(snaps) < 2:
                        continue
                    report = check_drift(cid)
                    if report.needs_reindex:
                        logger.warning(
                            "Background monitor: auto re-index triggered for '%s' "
                            "(composite=%.4f > threshold=%.4f)",
                            cid,
                            report.signals.composite_score,
                            report.threshold,
                        )
                        trigger_reindex(cid, trigger_reason="automatic_drift")
            except Exception as exc:
                logger.error("Drift monitor error: %s", exc)

    _monitor_thread = threading.Thread(
        target=_monitor_loop, daemon=True, name="drift-monitor"
    )
    _monitor_thread.start()


def stop_background_monitor() -> None:
    """Stop the background drift monitor thread."""
    _monitor_stop.set()
    logger.info("Drift monitor stopped.")