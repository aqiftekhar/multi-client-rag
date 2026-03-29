"""Unit tests for embedding drift detection."""

import pytest
import numpy as np
from app.evaluation.drift_detector import (
    record_snapshot,
    check_drift,
    get_drift_history,
    _snapshots,
)


def _random_embeddings(n: int, dim: int = 32) -> list[list[float]]:
    return np.random.randn(n, dim).tolist()


def _clear(client_id: str):
    _snapshots.pop(client_id, None)


class TestDriftDetector:
    def test_no_drift_with_single_snapshot(self):
        _clear("test_single")
        record_snapshot("test_single", _random_embeddings(50))
        report = check_drift("test_single")
        assert report.drift_score == 0.0
        assert not report.needs_reindex

    def test_no_drift_identical_snapshots(self):
        _clear("test_identical")
        emb = _random_embeddings(50)
        record_snapshot("test_identical", emb)
        record_snapshot("test_identical", emb)
        report = check_drift("test_identical")
        assert report.drift_score < 0.01  # near-zero drift

    def test_high_drift_detected(self):
        _clear("test_high_drift")
        # Two completely different embedding distributions
        np.random.seed(0)
        emb1 = (np.ones((50, 32)) + np.random.randn(50, 32) * 0.01).tolist()
        emb2 = (-np.ones((50, 32)) + np.random.randn(50, 32) * 0.01).tolist()
        record_snapshot("test_high_drift", emb1)
        record_snapshot("test_high_drift", emb2)
        report = check_drift("test_high_drift")
        assert report.drift_score > report.threshold
        assert report.needs_reindex

    def test_history_accumulates(self):
        _clear("test_history")
        for _ in range(4):
            record_snapshot("test_history", _random_embeddings(20))
        history = get_drift_history("test_history")
        assert len(history) == 3  # n snapshots → n-1 drift scores

    def test_max_snapshots_respected(self):
        _clear("test_maxsnap")
        for _ in range(15):  # exceed _MAX_SNAPSHOTS = 10
            record_snapshot("test_maxsnap", _random_embeddings(10))
        from app.evaluation.drift_detector import _get_snapshots
        assert len(_get_snapshots("test_maxsnap")) <= 10
