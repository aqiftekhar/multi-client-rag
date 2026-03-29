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
