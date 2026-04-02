"""Evaluation routes — metrics, drift, signals, and CI/CD pipeline."""

from fastapi import APIRouter, HTTPException

from app.api.models import (
    RunEvalRequest,
    EvalReportResponse,
    ClientEvalResultResponse,
    DriftReportResponse,
    SignalSummaryResponse,
)
from app.clients import manager
from app.evaluation.pipeline import run_eval_pipeline, EvalQuery
from app.evaluation.drift_detector import check_drift, get_drift_history
from app.evaluation.signals import summarize as signal_summary, get_signals

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run", response_model=EvalReportResponse)
def run_eval(req: RunEvalRequest) -> EvalReportResponse:
    """Run the CI/CD evaluation pipeline across one or more clients.

    This is the pre-deploy gate. Returns pass/fail per client and overall.
    """
    # Validate all clients exist
    for client_id in req.client_queries:
        if not manager.exists(client_id):
            raise HTTPException(
                status_code=404,
                detail=f"Client '{client_id}' not registered.",
            )

    # Convert request models to internal EvalQuery dataclasses
    client_queries = {
        cid: [EvalQuery(query=q.query, relevant_chunk_ids=q.relevant_chunk_ids) for q in queries]
        for cid, queries in req.client_queries.items()
    }

    report = run_eval_pipeline(client_queries, k=req.k)

    return EvalReportResponse(
        passed=report.passed,
        client_results=[
            ClientEvalResultResponse(
                client_id=r.client_id,
                num_queries=r.num_queries,
                metrics=r.metrics,
                drift_score=r.drift_score,
                signal_summary=r.signal_summary,
                passed=r.passed,
                failure_reasons=r.failure_reasons,
            )
            for r in report.client_results
        ],
        total_clients=report.total_clients,
        passed_clients=report.passed_clients,
        summary=report.summary,
    )


@router.get("/metrics/{client_id}")
def get_metrics(client_id: str) -> dict:
    """Return the latest signal summary (task completion, retry rate, etc.) for a client."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    return signal_summary(client_id)


@router.get("/drift/{client_id}", response_model=DriftReportResponse)
def get_drift(client_id: str) -> DriftReportResponse:
    """Return the current embedding drift report for a client."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    report = check_drift(client_id)
    history = get_drift_history(client_id)
    return DriftReportResponse(
        client_id=report.client_id,
        current_similarity=report.current_similarity,
        drift_score=report.drift_score,
        threshold=report.threshold,
        needs_reindex=report.needs_reindex,
        snapshot_count=report.snapshot_count,
        latest_snapshot_at=report.latest_snapshot_at,
        history=history,
    )


@router.get("/signals/{client_id}", response_model=SignalSummaryResponse)
def get_signals_route(client_id: str, limit: int = 50) -> SignalSummaryResponse:
    """Return implicit failure signals for a client."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    summary = signal_summary(client_id)
    recent = get_signals(client_id, limit=limit)
    return SignalSummaryResponse(
        client_id=summary["client_id"],
        total_signals=summary["total_signals"],
        counts=summary["counts"],
        task_completion_rate=summary["task_completion_rate"],
        retry_rate=summary["retry_rate"],
        failure_rate=summary["failure_rate"],
        recent_signals=[
            {
                "type": s.signal_type,
                "query": s.query[:80],
                "timestamp": s.timestamp,
                "details": s.details,
            }
            for s in recent
        ],
    )

@router.get("/hallucinations/{client_id}")
def get_hallucinations(client_id: str, limit: int = 50) -> dict:
    """Return recent hallucination records for a client.

    Use this to understand what is going wrong with retrieval.
    """
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    from app.evaluation.hallucination_log import get_records
    records = get_records(client_id=client_id, limit=limit)
    return {
        "client_id": client_id,
        "total": len(records),
        "records": records,
    }


@router.get("/hallucinations/{client_id}/analysis")
def analyze_hallucinations(client_id: str) -> dict:
    """Analyze hallucination patterns and return actionable recommendations.

    This is the Offline Analysis step in the improvement loop.
    Use the recommendations to improve retrieval, chunking, and prompts.
    """
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    from app.evaluation.hallucination_log import analyze
    return analyze(client_id=client_id)

@router.get("/hallucinations/{client_id}")
def get_hallucinations(client_id: str, limit: int = 50) -> dict:
    """Return recent hallucination records for a client."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    try:
        from app.evaluation.hallucination_log import get_records
        records = get_records(client_id=client_id, limit=limit)
        return {"client_id": client_id, "total": len(records), "records": records}
    except Exception:
        return {"client_id": client_id, "total": 0, "records": []}


@router.get("/hallucinations/{client_id}/analysis")
def analyze_hallucinations(client_id: str) -> dict:
    """Analyze hallucination patterns and return actionable recommendations."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    try:
        from app.evaluation.hallucination_log import analyze
        return analyze(client_id=client_id)
    except Exception:
        return {
            "total_hallucinations": 0,
            "message": "No hallucination data yet. Run some queries first.",
        }
    
#     @router.get("/drift/{client_id}/report")
# def get_drift_report(client_id: str) -> dict:
#     """Full drift analysis report with all three signals and history."""
#     if not manager.exists(client_id):
#         raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
#     from app.evaluation.drift_detector import check_drift
#     report = check_drift(client_id)
#     return {
#         "client_id": report.client_id,
#         "needs_reindex": report.needs_reindex,
#         "severity": report.severity,
#         "threshold": report.threshold,
#         "snapshot_count": report.snapshot_count,
#         "latest_snapshot_at": report.latest_snapshot_at,
#         "recommendation": report.recommendation,
#         "signals": {
#             "centroid_similarity": report.signals.centroid_similarity,
#             "centroid_drift": report.signals.centroid_drift,
#             "variance_drift": report.signals.variance_drift,
#             "js_divergence": report.signals.js_divergence,
#             "composite_score": report.signals.composite_score,
#             "n_chunks_before": report.signals.n_chunks_before,
#             "n_chunks_after": report.signals.n_chunks_after,
#             "chunks_delta": report.signals.chunks_delta,
#         },
#         "history": report.history,
#     }


# @router.post("/drift/{client_id}/reindex")
# def manual_reindex(client_id: str) -> dict:
#     """Manually trigger re-indexing for a client."""
#     if not manager.exists(client_id):
#         raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
#     from app.evaluation.drift_detector import trigger_reindex
#     event = trigger_reindex(client_id, trigger_reason="manual")
#     return {
#         "event_id": event.event_id,
#         "client_id": event.client_id,
#         "status": event.status,
#         "trigger_reason": event.trigger_reason,
#         "drift_score_before": event.drift_score_before,
#         "drift_score_after": event.drift_score_after,
#         "chunks_before": event.chunks_before,
#         "chunks_after": event.chunks_after,
#         "duration_seconds": event.duration_seconds,
#         "error": event.error,
#     }


# @router.get("/drift/{client_id}/reindex-log")
# def get_reindex_log(client_id: str) -> dict:
#     """Return all re-index events for a client."""
#     if not manager.exists(client_id):
#         raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
#     from app.evaluation.drift_detector import get_reindex_log
#     events = get_reindex_log(client_id)
#     return {
#         "client_id": client_id,
#         "total_reindex_events": len(events),
#         "events": [e.to_dict() for e in reversed(events)],  # newest first
#     }

@router.get("/drift/{client_id}/report")
def get_drift_report(client_id: str) -> dict:
    """Full drift analysis report with all three signals and history."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    from app.evaluation.drift_detector import check_drift
    report = check_drift(client_id)
    return {
        "client_id": report.client_id,
        "needs_reindex": report.needs_reindex,
        "severity": report.severity,
        "threshold": report.threshold,
        "snapshot_count": report.snapshot_count,
        "latest_snapshot_at": report.latest_snapshot_at,
        "recommendation": report.recommendation,
        "signals": {
            "centroid_similarity": report.signals.centroid_similarity,
            "centroid_drift": report.signals.centroid_drift,
            "variance_drift": report.signals.variance_drift,
            "js_divergence": report.signals.js_divergence,
            "composite_score": report.signals.composite_score,
            "n_chunks_before": report.signals.n_chunks_before,
            "n_chunks_after": report.signals.n_chunks_after,
            "chunks_delta": report.signals.chunks_delta,
        },
        "history": report.history,
    }


@router.post("/drift/{client_id}/reindex")
def manual_reindex(client_id: str) -> dict:
    """Manually trigger re-indexing for a client."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    from app.evaluation.drift_detector import trigger_reindex
    event = trigger_reindex(client_id, trigger_reason="manual")
    return {
        "event_id": event.event_id,
        "client_id": event.client_id,
        "status": event.status,
        "trigger_reason": event.trigger_reason,
        "drift_score_before": event.drift_score_before,
        "drift_score_after": event.drift_score_after,
        "chunks_before": event.chunks_before,
        "chunks_after": event.chunks_after,
        "duration_seconds": event.duration_seconds,
        "error": event.error,
    }


@router.get("/drift/{client_id}/reindex-log")
def get_reindex_log(client_id: str) -> dict:
    """Return all re-index events for a client."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    from app.evaluation.drift_detector import get_reindex_log
    events = get_reindex_log(client_id)
    return {
        "client_id": client_id,
        "total_reindex_events": len(events),
        "events": [e.to_dict() for e in reversed(events)],  # newest first
    }

## The complete picture
"""
User Query
    ↓
RetrievalAgent
    normal:          coarse_k=20 → cross-encoder top-5
    after hallucination: coarse_k=40 → cross-encoder top-5  ← retry_retrieval()
    ↓
LLM call
    normal:          standard prompt
    strict_mode=True: + STRICT MODE instruction              ← regenerate_answer(strict_mode)
    ↓
ValidationAgent
    schema check
    source check     → filename matching
    faithfulness     → cross-encoder scores answer vs chunks
                       < -2.0 = CONTAIN (clear answer, strict_mode=True)  ← contain
                                                                           ← verify
    ↓
CorrectionAgent
    hallucination    → tighten query, retry              ← recover
    low_confidence   → reformulate, retry
    retrieval_fail   → simplify, retry
    max retries hit  → LOG EVERYTHING                    ← improve
                       hallucination → safe fallback
                       low_confidence → ask clarification
    ↓
EvaluationAgent
    success with recovery → LOG with recovery_succeeded=True
    ↓
data/hallucination_log.jsonl
    ↓
GET /eval/hallucinations/{client_id}/analysis
    → which queries hallucinate most
    → which documents are problematic
    → recovery rate
    → actionable recommendations    ← Offline Analysis → Deploy Better Version
"""