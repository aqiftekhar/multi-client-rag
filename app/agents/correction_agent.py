"""Correction agent — reformulates queries on failure and triggers retries."""

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.evaluation.signals import record, SignalType


_REFORMULATION_SUFFIXES = [
    " Please be specific and cite the source documents.",
    " Explain step by step using only the provided context.",
    " Summarise the key points relevant to this question.",
]


class CorrectionAgent(BaseAgent):
    """Reformulates the query when validation or retrieval fails."""

    name = "correction_agent"

    def run(self, context: AgentContext) -> AgentResult:
        record(
            SignalType.AGENT_RETRY,
            client_id=context.client_id,
            query=context.query,
            details={
                "retry_count": context.retry_count,
                "errors": context.errors,
                "run_id": context.run_id,
            },
        )

        if context.retry_count >= context.max_retries:
            self.logger.warning(
                "Max retries (%d) reached for run '%s'.", context.max_retries, context.run_id
            )
            # Provide a graceful fallback answer
            context.final_answer = (
                "I could not find a sufficiently reliable answer in the available documents. "
                "Please rephrase your question or check that relevant documents have been ingested."
            )
            context.confidence = 0.0
            return AgentResult(
                success=False,
                updated_context=context,
                message="max_retries_reached",
                should_retry=False,
            )

        # Reformulate the query
        suffix = _REFORMULATION_SUFFIXES[
            context.retry_count % len(_REFORMULATION_SUFFIXES)
        ]
        original_query = context.query
        context.query = context.query.rstrip(".") + suffix
        context.retry_count += 1
        context.errors = []  # reset for the next attempt

        self.logger.info(
            "Query reformulated (attempt %d): '%s' → '%s'",
            context.retry_count,
            original_query[:60],
            context.query[:60],
        )

        return AgentResult(
            success=True,
            updated_context=context,
            message=f"query_reformulated_attempt_{context.retry_count}",
            should_retry=True,
        )
