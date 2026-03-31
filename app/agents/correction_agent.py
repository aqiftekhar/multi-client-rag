"""Correction agent — recover and improve steps.

Also writes the complete hallucination record to the structured log
so every event feeds the continuous improvement loop.
"""

import logging
from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.evaluation.signals import record, SignalType

logger = logging.getLogger(__name__)

_SAFE_FALLBACK = (
    "I was unable to generate a reliable answer grounded in your documents. "
    "\n\nThis usually means the retrieved context did not contain enough "
    "information to answer this question confidently."
    "\n\nSuggestions:"
    "\n• Try rephrasing your question with more specific terms"
    "\n• Check that the relevant documents have been ingested"
    "\n• Break your question into smaller, more focused parts"
)

_CLARIFICATION_REQUEST = (
    "I found some relevant information but could not generate a confident answer. "
    "Could you clarify:"
    "\n• What specific aspect are you asking about?"
    "\n• Are you referring to a specific document, section, or time period?"
    "\n• What would a helpful answer look like for your use case?"
)


class CorrectionAgent(BaseAgent):
    name = "correction_agent"

    def run(self, context: AgentContext) -> AgentResult:
        record(SignalType.AGENT_RETRY, client_id=context.client_id,
               query=context.query,
               details={"retry_count": context.retry_count,
                        "retry_reason": context.retry_reason,
                        "strict_mode": context.strict_mode,
                        "run_id": context.run_id})

        if context.retry_count >= context.max_retries:
            return self._improve(context)

        if context.retry_reason == "hallucination":
            return self._recover_hallucination(context)

        if context.retry_reason == "low_confidence":
            return self._recover_low_confidence(context)

        return self._recover_retrieval_failure(context)

    def _recover_hallucination(self, context: AgentContext) -> AgentResult:
        """recover step — retry_retrieval() + regenerate_answer(strict_mode=True)."""
        context.retry_count += 1
        context.query = (
            context.query.rstrip(".")
            + ". Use ONLY information explicitly stated in the provided context."
        )
        # strict_mode already True → RetrievalAgent doubles coarse_k
        # Orchestrator injects STRICT MODE instruction into LLM prompt
        context.errors = []
        logger.info("Hallucination recovery attempt %d — run='%s'",
                    context.retry_count, context.run_id)
        return AgentResult(success=True, updated_context=context,
                           message=f"hallucination_recovery_{context.retry_count}",
                           should_retry=True)

    def _recover_low_confidence(self, context: AgentContext) -> AgentResult:
        """recover step — reformulate with more specificity."""
        context.retry_count += 1
        suffixes = [
            " Please be specific and cite the source documents.",
            " Focus only on what the documents explicitly state.",
            " Summarise the most directly relevant points from the context.",
        ]
        context.query = (context.query.rstrip(".")
                         + suffixes[context.retry_count % len(suffixes)])
        context.errors = []
        logger.info("Low confidence recovery attempt %d — run='%s'",
                    context.retry_count, context.run_id)
        return AgentResult(success=True, updated_context=context,
                           message=f"low_confidence_recovery_{context.retry_count}",
                           should_retry=True)

    def _recover_retrieval_failure(self, context: AgentContext) -> AgentResult:
        """recover step — simplify the query."""
        context.retry_count += 1
        words = context.query.split()
        if len(words) > 8:
            context.query = " ".join(words[:6]) + "?"
        else:
            context.query = context.query.rstrip(".") + " Provide a brief answer."
        context.errors = []
        logger.info("Retrieval failure recovery attempt %d — run='%s'",
                    context.retry_count, context.run_id)
        return AgentResult(success=True, updated_context=context,
                           message=f"retrieval_recovery_{context.retry_count}",
                           should_retry=True)

    def _improve(self, context: AgentContext) -> AgentResult:
        """improve step — max retries hit.

        Also writes a structured hallucination record to the improvement log.
        This is the LOG EVERYTHING step in the continuous improvement loop.
        """
        logger.warning("Max retries hit — run='%s' reason='%s'. Entering improve step.",
                       context.max_retries, context.run_id)

        # Determine final response
        if context.retry_reason == "low_confidence":
            context.final_answer = _CLARIFICATION_REQUEST
            recovery_strategy = "clarification_requested"
            recovery_succeeded = False
        else:
            context.final_answer = _SAFE_FALLBACK
            recovery_strategy = "safe_fallback"
            recovery_succeeded = False

        context.confidence = 0.0

        # LOG EVERYTHING — write structured hallucination record
        self._log_to_improvement_loop(context, recovery_strategy, recovery_succeeded)

        return AgentResult(success=False, updated_context=context,
                           message=f"improve_{recovery_strategy}",
                           should_retry=False)

    def _log_to_improvement_loop(
        self,
        context: AgentContext,
        recovery_strategy: str,
        recovery_succeeded: bool,
    ) -> None:
        """Write a complete structured record for offline analysis."""
        try:
            from app.evaluation.hallucination_log import log_hallucination, HallucinationRecord
            top_chunk_preview = ""
            retrieved_sources = []
            if context.retrieved_chunks:
                top = context.retrieved_chunks[0]
                top_chunk_preview = top.metadata.get("raw_text", top.text)[:200]
                retrieved_sources = list({c.source for c in context.retrieved_chunks})

            log_hallucination(HallucinationRecord(
                run_id=context.run_id,
                client_id=context.client_id,
                timestamp=__import__('datetime').datetime.now(
                    __import__('datetime').timezone.utc).isoformat(),
                original_query=context.query,
                final_query=context.query,
                retrieved_sources=retrieved_sources,
                retrieved_chunk_count=len(context.retrieved_chunks),
                top_chunk_preview=top_chunk_preview,
                llm_raw_output=context.llm_raw_output or "",
                faithfulness_score=context.hallucination_faithfulness_score,
                hallucination_type=context.hallucination_type or context.retry_reason,
                unmatched_citations=context.unmatched_citations,
                retry_count=context.retry_count,
                recovery_strategy=recovery_strategy,
                final_answer=context.final_answer,
                recovery_succeeded=recovery_succeeded,
            ))
        except Exception as exc:
            logger.error("Failed to write hallucination log record: %s", exc)