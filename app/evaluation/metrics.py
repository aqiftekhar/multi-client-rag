"""Pure evaluation metric functions — no side effects, fully testable.

Implements: Recall@k, Precision@k, MRR, NDCG.
"""

import math


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved results.

    Args:
        relevant: Set of relevant chunk/doc IDs (ground truth).
        retrieved: Ordered list of retrieved IDs (most relevant first).
        k: Cut-off rank.

    Returns:
        Float in [0, 1].
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant.

    Args:
        relevant: Set of relevant IDs.
        retrieved: Ordered list of retrieved IDs.
        k: Cut-off rank.

    Returns:
        Float in [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    """Reciprocal of the rank of the first relevant result.

    Returns 0 if no relevant result is found.
    """
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(queries: list[tuple[set[str], list[str]]]) -> float:
    """Mean Reciprocal Rank over a list of (relevant, retrieved) query pairs."""
    if not queries:
        return 0.0
    return sum(reciprocal_rank(rel, ret) for rel, ret in queries) / len(queries)


def dcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], 1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Normalized DCG at rank k.

    Normalises by the ideal DCG (all relevant items at the top).
    """
    actual_dcg = dcg_at_k(relevant, retrieved, k)
    # Ideal: all relevant items ranked first
    ideal_retrieved = list(relevant) + [f"__non_relevant_{i}" for i in range(k)]
    ideal_dcg = dcg_at_k(relevant, ideal_retrieved, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_all(
    relevant: set[str],
    retrieved: list[str],
    k: int = 5,
) -> dict[str, float]:
    """Compute all retrieval metrics in one call.

    Returns a dict with keys: recall@k, precision@k, mrr, ndcg@k.
    """
    return {
        f"recall@{k}": recall_at_k(relevant, retrieved, k),
        f"precision@{k}": precision_at_k(relevant, retrieved, k),
        "mrr": reciprocal_rank(relevant, retrieved),
        f"ndcg@{k}": ndcg_at_k(relevant, retrieved, k),
    }
