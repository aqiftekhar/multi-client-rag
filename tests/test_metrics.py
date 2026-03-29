"""Unit tests for evaluation metrics — pure functions, no LLM or DB required."""

import pytest
from app.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    compute_all,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k({"a", "b", "c"}, ["a", "b", "c", "d", "e"], k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k({"x", "y"}, ["a", "b", "c"], k=3) == 0.0

    def test_partial_recall(self):
        assert recall_at_k({"a", "b"}, ["a", "c", "d"], k=3) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(set(), ["a", "b"], k=2) == 0.0

    def test_k_limits_results(self):
        # "b" is at rank 2 but k=1, so only "a" counts
        assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=1) == 0.5


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k({"a", "b", "c"}, ["a", "b", "c"], k=3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k({"x"}, ["a", "b", "c"], k=3) == 0.0

    def test_half_precision(self):
        assert precision_at_k({"a", "c"}, ["a", "b", "c", "d"], k=4) == 0.5

    def test_k_zero(self):
        assert precision_at_k({"a"}, ["a"], k=0) == 0.0


class TestReciprocalRank:
    def test_first_rank(self):
        assert reciprocal_rank({"a"}, ["a", "b", "c"]) == 1.0

    def test_second_rank(self):
        assert reciprocal_rank({"b"}, ["a", "b", "c"]) == 0.5

    def test_third_rank(self):
        assert abs(reciprocal_rank({"c"}, ["a", "b", "c"]) - 1/3) < 1e-9

    def test_not_found(self):
        assert reciprocal_rank({"z"}, ["a", "b", "c"]) == 0.0


class TestMRR:
    def test_all_first(self):
        queries = [
            ({"a"}, ["a", "b"]),
            ({"c"}, ["c", "d"]),
        ]
        assert mean_reciprocal_rank(queries) == 1.0

    def test_mixed(self):
        queries = [
            ({"a"}, ["a", "b"]),   # RR = 1.0
            ({"b"}, ["a", "b"]),   # RR = 0.5
        ]
        assert mean_reciprocal_rank(queries) == 0.75

    def test_empty(self):
        assert mean_reciprocal_rank([]) == 0.0


class TestNDCG:
    def test_perfect(self):
        assert ndcg_at_k({"a", "b"}, ["a", "b", "c", "d"], k=4) == 1.0

    def test_zero(self):
        assert ndcg_at_k({"x", "y"}, ["a", "b", "c"], k=3) == 0.0

    def test_order_matters(self):
        # Having relevant item first should score higher than having it second
        ndcg_first = ndcg_at_k({"a"}, ["a", "b", "c"], k=3)
        ndcg_second = ndcg_at_k({"b"}, ["a", "b", "c"], k=3)
        assert ndcg_first > ndcg_second


class TestComputeAll:
    def test_returns_all_keys(self):
        result = compute_all({"a", "b"}, ["a", "b", "c", "d", "e"], k=5)
        assert "recall@5" in result
        assert "precision@5" in result
        assert "mrr" in result
        assert "ndcg@5" in result

    def test_all_floats(self):
        result = compute_all({"a"}, ["a", "b"], k=2)
        for v in result.values():
            assert isinstance(v, float)
