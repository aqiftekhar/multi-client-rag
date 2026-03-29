"""Evaluation agent — records outcome signals and updates drift snapshots."""

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.evaluation.signals import record, SignalType
from app.evaluation.drift_detector import record_snapshot
from app.rag.embedder import embed_texts


class EvaluationAgent(BaseAgent):
    """Records outcome signals and refreshes drift snapshots after ingestion."""

    name = "evaluation_agent"

    def run(self, context: AgentContext) -> AgentResult:
        # Determine outcome
        if context.final_answer and context.confidence >= 0.4:
            record(
                SignalType.TASK_SUCCESS,
                client_id=context.client_id,
                query=context.query,
                details={
                    "confidence": context.confidence,
                    "chunks_used": len(context.pruned_chunks),
                    "run_id": context.run_id,
                    "sources": context.sources_used,
                },
            )
        else:
            record(
                SignalType.TASK_FAILURE,
                client_id=context.client_id,
                query=context.query,
                details={
                    "confidence": context.confidence,
                    "errors": context.errors,
                    "run_id": context.run_id,
                },
            )

        self.logger.debug(
            "Eval agent: run_id=%s success=%s confidence=%.2f",
            context.run_id,
            bool(context.final_answer),
            context.confidence,
        )

        return AgentResult(
            success=True,
            updated_context=context,
            message="evaluation_recorded",
        )


def update_drift_snapshot(client_id: str, chunk_texts: list[str]) -> None:
    """Recompute and store a drift snapshot for *client_id*.

    Should be called after bulk ingestion.
    """
    if not chunk_texts:
        return
    embeddings = embed_texts(chunk_texts)
    record_snapshot(client_id, embeddings)
