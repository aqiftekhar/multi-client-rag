"""Implicit failure signal store — append-only log of agent events.

Signals are persisted to disk so they survive app restarts.
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

_SIGNALS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "signals"
)


class SignalType(str, Enum):
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    VALIDATION_FAILURE = "validation_failure"
    CORRECTION_TRIGGERED = "correction_triggered"
    AGENT_RETRY = "agent_retry"
    HALLUCINATION_DETECTED = "hallucination_detected"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class Signal:
    """A single implicit signal event."""
    signal_type: SignalType
    client_id: str
    query: str
    timestamp: str
    details: dict = field(default_factory=dict)


# Global append-only store keyed by client_id
_store: dict[str, list[Signal]] = defaultdict(list)


# ── Persistence ───────────────────────────────────────────────────────────────

def _signal_file(client_id: str) -> str:
    safe = client_id.replace("/", "_")
    return os.path.join(_SIGNALS_DIR, f"{safe}.json")


def _persist(client_id: str) -> None:
    """Save signals for one client to disk."""
    try:
        os.makedirs(_SIGNALS_DIR, exist_ok=True)
        signals = _store[client_id]
        # Keep last 10000 signals per client to avoid unbounded growth
        signals_to_save = signals[-10000:]
        with open(_signal_file(client_id), "w") as f:
            json.dump(
                [
                    {
                        "signal_type": s.signal_type,
                        "client_id": s.client_id,
                        "query": s.query,
                        "timestamp": s.timestamp,
                        "details": s.details,
                    }
                    for s in signals_to_save
                ],
                f,
                indent=2,
            )
    except Exception as exc:
        logger.error("Failed to persist signals for '%s': %s", client_id, exc)


def load_signals_from_disk() -> None:
    """Load all persisted signals on startup.

    Call this from app lifespan before serving requests.
    """
    if not os.path.exists(_SIGNALS_DIR):
        logger.info("No signals directory found — starting fresh.")
        return
    loaded = 0
    for fname in os.listdir(_SIGNALS_DIR):
        if not fname.endswith(".json"):
            continue
        client_id = fname[:-5]
        try:
            with open(os.path.join(_SIGNALS_DIR, fname)) as f:
                data = json.load(f)
            _store[client_id] = [
                Signal(
                    signal_type=s["signal_type"],
                    client_id=s["client_id"],
                    query=s["query"],
                    timestamp=s["timestamp"],
                    details=s.get("details", {}),
                )
                for s in data
            ]
            loaded += len(_store[client_id])
        except Exception as exc:
            logger.error("Failed to load signals for '%s': %s", client_id, exc)
    logger.info("Loaded %d signals from disk across all clients.", loaded)


# ── Public API ────────────────────────────────────────────────────────────────

def record(
    signal_type: SignalType,
    client_id: str,
    query: str,
    details: dict | None = None,
) -> Signal:
    """Record a signal event and persist to disk. Returns the created signal."""
    sig = Signal(
        signal_type=signal_type,
        client_id=client_id,
        query=query,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details=details or {},
    )
    _store[client_id].append(sig)
    _persist(client_id)
    logger.debug(
        "Signal recorded: %s | client=%s | query=%.60s",
        signal_type, client_id, query,
    )
    return sig


def get_signals(client_id: str, limit: int = 100) -> list[Signal]:
    """Return the most recent *limit* signals for *client_id*."""
    return list(_store[client_id])[-limit:]


def summarize(client_id: str) -> dict:
    """Return a summary of signal counts by type for *client_id*."""
    signals = _store[client_id]
    counts: dict[str, int] = defaultdict(int)
    for sig in signals:
        counts[sig.signal_type] += 1

    total = len(signals)
    successes = counts.get(SignalType.TASK_SUCCESS, 0)
    failures = (
        counts.get(SignalType.TASK_FAILURE, 0)
        + counts.get(SignalType.VALIDATION_FAILURE, 0)
    )
    retries = counts.get(SignalType.AGENT_RETRY, 0)

    terminal_events = successes + failures
    task_completion_rate = (
        successes / terminal_events if terminal_events > 0 else 0.0
    )
    retry_rate = retries / total if total > 0 else 0.0

    return {
        "client_id": client_id,
        "total_signals": total,
        "counts": dict(counts),
        "task_completion_rate": round(task_completion_rate, 4),
        "retry_rate": round(retry_rate, 4),
        "failure_rate": round(failures / terminal_events, 4) if terminal_events > 0 else 0.0,
    }