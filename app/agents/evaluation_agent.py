"""Evaluation agent — records outcome signals and updates drift snapshots."""

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.evaluation.signals import record, SignalType
from app.evaluation.drift_detector import record_snapshot
from app.rag.embedder import embed_texts


class EvaluationAgent(BaseAgent):
    """Records outcome signals and refreshes drift snapshots after ingestion."""

    name = "evaluation_agent"

    def run(self, context: AgentContext) -> AgentResult:
        """Record outcome signals.

        NOT FOUND answers are task_success — the system correctly identified
        that the document does not contain the answer. This is not a failure.

        task_failure is reserved for cases where the system tried to answer
        a question it could answer but produced an unreliable result.
        """
        is_not_found = (
            context.retry_reason == "not_answerable"
            or (context.final_answer or "").strip().startswith("NOT FOUND")
        )

        is_negative_denial = any(
            phrase in (context.final_answer or "").lower()
            for phrase in [
                "does not mention",
                "does not use",
                "does not describe",
                "not present in",
                "not found",
                "explicitly states it does not",
                "explicitly contradicts",
            ]
        )

        # Determine if this is a successful outcome
        # Success = answered correctly OR correctly identified as not answerable
        is_success = (
            (context.final_answer and context.confidence >= 0.4)
            or is_not_found
            or is_negative_denial
        )

        if is_success:
            record(
                SignalType.TASK_SUCCESS,
                client_id=context.client_id,
                query=context.query,
                details={
                    "confidence": context.confidence,
                    "chunks_used": len(context.pruned_chunks),
                    "run_id": context.run_id,
                    "sources": context.sources_used,
                    "outcome_type": (
                        "not_found" if is_not_found
                        else "negative_denial" if is_negative_denial
                        else "answered"
                    ),
                    "had_hallucination_recovery": context.retry_reason == "hallucination",
                },
            )

            # Log recovered hallucinations to improvement loop
            if context.retry_reason == "hallucination" and context.retry_count > 0:
                try:
                    from app.evaluation.hallucination_log import (
                        log_hallucination, HallucinationRecord,
                    )
                    import datetime
                    retrieved_sources = list({c.source for c in context.retrieved_chunks})
                    top_chunk_preview = ""
                    if context.retrieved_chunks:
                        top = context.retrieved_chunks[0]
                        top_chunk_preview = top.metadata.get("raw_text", top.text)[:200]

                    log_hallucination(HallucinationRecord(
                        run_id=context.run_id,
                        client_id=context.client_id,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc).isoformat(),
                        original_query=context.original_query or context.query,
                        final_query=context.query,
                        retrieved_sources=retrieved_sources,
                        retrieved_chunk_count=len(context.retrieved_chunks),
                        top_chunk_preview=top_chunk_preview,
                        llm_raw_output=context.llm_raw_output or "",
                        faithfulness_score=context.hallucination_faithfulness_score,
                        hallucination_type=context.hallucination_type or "recovered",
                        unmatched_citations=context.unmatched_citations,
                        retry_count=context.retry_count,
                        recovery_strategy="hallucination_recovery",
                        final_answer=context.final_answer,
                        recovery_succeeded=True,
                    ))
                except Exception as exc:
                    self.logger.error("Failed to log recovered hallucination: %s", exc)

        else:
            # True failure — system tried but could not produce reliable answer
            record(
                SignalType.TASK_FAILURE,
                client_id=context.client_id,
                query=context.query,
                details={
                    "confidence": context.confidence,
                    "errors": context.errors,
                    "run_id": context.run_id,
                    "retry_reason": context.retry_reason,
                },
            )

        self.logger.debug(
            "Eval: run_id=%s success=%s confidence=%.2f outcome=%s",
            context.run_id,
            is_success,
            context.confidence,
            "not_found" if is_not_found else "negative_denial" if is_negative_denial else "answered",
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
