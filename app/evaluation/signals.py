"""Implicit failure signal store — append-only log of agent events.

Signals are the ground truth for evaluation: retries, corrections,
validation failures, and task completions are all recorded here.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


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


def record(
    signal_type: SignalType,
    client_id: str,
    query: str,
    details: dict | None = None,
) -> Signal:
    """Record a signal event. Returns the created signal."""
    sig = Signal(
        signal_type=signal_type,
        client_id=client_id,
        query=query,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details=details or {},
    )
    _store[client_id].append(sig)
    logger.debug("Signal recorded: %s | client=%s | query=%.60s", signal_type, client_id, query)
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
    failures = counts.get(SignalType.TASK_FAILURE, 0) + counts.get(SignalType.VALIDATION_FAILURE, 0)
    retries = counts.get(SignalType.AGENT_RETRY, 0)

    task_completion_rate = successes / total if total > 0 else 0.0
    retry_rate = retries / total if total > 0 else 0.0

    return {
        "client_id": client_id,
        "total_signals": total,
        "counts": dict(counts),
        "task_completion_rate": round(task_completion_rate, 4),
        "retry_rate": round(retry_rate, 4),
        "failure_rate": round(failures / total, 4) if total > 0 else 0.0,
    }
