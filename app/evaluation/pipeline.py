"""CI/CD evaluation pipeline — runs per-client, parallel, before deploys."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from app.config import get_settings
from app.evaluation.metrics import compute_all, mean_reciprocal_rank
from app.evaluation.drift_detector import check_drift
from app.evaluation.signals import summarize as signal_summary
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """A single evaluation query with ground-truth relevant chunk IDs."""

    query: str
    relevant_chunk_ids: list[str]   # ground-truth IDs that should be retrieved


@dataclass
class ClientEvalResult:
    """Evaluation results for a single client."""

    client_id: str
    num_queries: int
    metrics: dict[str, float]
    drift_score: float
    signal_summary: dict
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated evaluation report across all clients."""

    passed: bool
    client_results: list[ClientEvalResult]
    total_clients: int
    passed_clients: int
    summary: dict


def _eval_client(client_id: str, queries: list[EvalQuery], k: int = 5) -> ClientEvalResult:
    """Run evaluation for a single client."""
    cfg = get_settings()
    all_metrics: list[dict] = []
    mrr_pairs: list[tuple[set[str], list[str]]] = []
    failure_reasons: list[str] = []

    for eq in queries:
        retrieved_chunks = retrieve(eq.query, client_id=client_id, fine_k=k)
        retrieved_ids = [c.chunk_id for c in retrieved_chunks]
        relevant = set(eq.relevant_chunk_ids)

        metrics = compute_all(relevant, retrieved_ids, k=k)
        all_metrics.append(metrics)
        mrr_pairs.append((relevant, retrieved_ids))

    # Average metrics across queries
    if all_metrics:
        avg_metrics: dict[str, float] = {}
        for key in all_metrics[0]:
            avg_metrics[key] = round(
                sum(m[key] for m in all_metrics) / len(all_metrics), 4
            )
        avg_metrics["mrr"] = round(mean_reciprocal_rank(mrr_pairs), 4)
    else:
        avg_metrics = {f"recall@{k}": 0.0, f"precision@{k}": 0.0, "mrr": 0.0, f"ndcg@{k}": 0.0}

    # Drift check
    drift_report = check_drift(client_id)

    # Signal summary
    sig_summary = signal_summary(client_id)

    # Pass/fail decision
    recall_key = f"recall@{k}"
    ndcg_key = f"ndcg@{k}"
    passed = True

    if avg_metrics.get(recall_key, 0) < cfg.eval_recall_min:
        failure_reasons.append(
            f"{recall_key}={avg_metrics[recall_key]:.3f} < threshold {cfg.eval_recall_min}"
        )
        passed = False

    if avg_metrics.get(ndcg_key, 0) < cfg.eval_ndcg_min:
        failure_reasons.append(
            f"{ndcg_key}={avg_metrics[ndcg_key]:.3f} < threshold {cfg.eval_ndcg_min}"
        )
        passed = False

    if drift_report.needs_reindex:
        failure_reasons.append(
            f"embedding_drift={drift_report.drift_score:.4f} > threshold {drift_report.threshold}"
        )
        # Drift is a warning — doesn't block deploy on its own
        # passed = False  # uncomment to make drift a hard failure

    logger.info(
        "Eval client '%s': passed=%s metrics=%s drift=%.4f",
        client_id,
        passed,
        avg_metrics,
        drift_report.drift_score,
    )

    return ClientEvalResult(
        client_id=client_id,
        num_queries=len(queries),
        metrics=avg_metrics,
        drift_score=drift_report.drift_score,
        signal_summary=sig_summary,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def run_eval_pipeline(
    client_queries: dict[str, list[EvalQuery]],
    k: int = 5,
    max_workers: int = 4,
) -> EvalReport:
    """Run evaluation across all clients in parallel.

    Args:
        client_queries: Mapping of client_id → list of EvalQuery.
        k: Rank cut-off for all metrics.
        max_workers: Thread pool size for parallel execution.
    """
    client_results: list[ClientEvalResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_eval_client, cid, queries, k): cid
            for cid, queries in client_queries.items()
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
                client_results.append(result)
            except Exception as exc:
                logger.error("Eval failed for client '%s': %s", cid, exc)
                client_results.append(
                    ClientEvalResult(
                        client_id=cid,
                        num_queries=len(client_queries[cid]),
                        metrics={},
                        drift_score=0.0,
                        signal_summary={},
                        passed=False,
                        failure_reasons=[f"eval_exception: {exc}"],
                    )
                )

    passed_clients = sum(1 for r in client_results if r.passed)
    overall_passed = passed_clients == len(client_results)

    summary = {
        "total_clients": len(client_results),
        "passed_clients": passed_clients,
        "overall_passed": overall_passed,
        "avg_metrics": _aggregate_metrics(client_results),
    }

    return EvalReport(
        passed=overall_passed,
        client_results=client_results,
        total_clients=len(client_results),
        passed_clients=passed_clients,
        summary=summary,
    )


def _aggregate_metrics(results: list[ClientEvalResult]) -> dict[str, float]:
    """Average metrics across all clients."""
    if not results:
        return {}
    all_keys = set()
    for r in results:
        all_keys.update(r.metrics.keys())
    agg = {}
    for key in all_keys:
        vals = [r.metrics[key] for r in results if key in r.metrics]
        agg[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return agg
