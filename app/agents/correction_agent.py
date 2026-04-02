"""Correction agent — non-deterministic escalating retry strategy.

Retry #1 (hallucination/not_answerable):
  → Widen retrieval (coarse_k doubled, already handled by RetrievalAgent)
  → Rewrite query to be more specific and grounded

Retry #2 (hallucination still failing):
  → Strict grounding prompt active (strict_mode=True)
  → Query rewritten to explicitly ask for NOT FOUND if absent

Retry #3 / max retries:
  → Safe fallback or clarification request

Why non-deterministic matters:
  Same query + same retrieval + same temperature = same wrong answer.
  Each retry must change at least one of: query, retrieval scope, or prompt constraints.
"""

import logging
from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.evaluation.signals import record, SignalType

logger = logging.getLogger(__name__)

_SAFE_FALLBACK = (
    "The provided documents do not contain enough information to answer "
    "this question reliably."
    "\n\nSuggestions:"
    "\n• Rephrase your question with more specific terms"
    "\n• Check that the relevant documents have been ingested"
    "\n• Break your question into smaller, more focused parts"
)

_CLARIFICATION_REQUEST = (
    "I found some relevant information but could not generate a confident answer. "
    "Could you clarify:"
    "\n• What specific aspect are you asking about?"
    "\n• Are you referring to a specific document, section, or time period?"
)

_NOT_FOUND_RESPONSE = (
    "This information is not present in the provided documents. "
    "The documents may explicitly contradict your question's assumption, "
    "or this topic may not be covered in the ingested content."
)


class CorrectionAgent(BaseAgent):
    name = "correction_agent"

    def run(self, context: AgentContext) -> AgentResult:
        record(
            SignalType.AGENT_RETRY,
            client_id=context.client_id,
            query=context.query,
            details={
                "retry_count": context.retry_count,
                "retry_reason": context.retry_reason,
                "strict_mode": context.strict_mode,
                "run_id": context.run_id,
            },
        )

        # Max retries → improve step
        if context.retry_count >= context.max_retries:
            return self._improve(context)

        # not_answerable goes straight to improve — never retry
        # The document does not have the answer; changing the prompt won't help
        if context.retry_reason == "not_answerable":
            return self._improve(context)

        if context.retry_reason == "hallucination":
            return self._escalating_retry(context)

        if context.retry_reason == "low_confidence":
            return self._recover_low_confidence(context)

        return self._recover_retrieval_failure(context)

    def _escalating_retry(self, context: AgentContext) -> AgentResult:
        """Non-deterministic escalating retry.

        Retry 1: Force the model to address the premise explicitly
        Retry 2: Hard NOT FOUND enforcement with exact quote requirement
        """
        context.retry_count += 1
        context.errors = []
        context.llm_raw_output = ""
        context.validated_response = None

        original = context.original_query or context.query.split(".")[0].split("?")[0]

        if context.retry_count == 1:
            # Force explicit premise resolution
            context.query = (
                f"{original.rstrip('?.')}? "
                "Before answering, explicitly state: does this subject "
                "actually exist as a component in the document, or does the document "
                "only mention it as an analogy? If it does not exist as a component, "
                "say so clearly first."
            )
            logger.info(
                "Escalating retry #1 — run='%s' forcing explicit premise resolution",
                context.run_id,
            )

        elif context.retry_count == 2:
            # Hard NOT FOUND enforcement
            context.query = (
                f"Regarding: {original.rstrip('?.')}. "
                "Check only if the document explicitly describes this as an actual "
                "component or mechanism. If the document only uses it as a comparison "
                "or analogy for something else, or if the document says it is not used, "
                "your answer MUST be: NOT FOUND or NOT USED in this document. "
                "Do not describe related components as substitutes."
            )
            context.strict_mode = True
            logger.info(
                "Escalating retry #2 — run='%s' hard NOT FOUND enforcement",
                context.run_id,
            )

        return AgentResult(
            success=True,
            updated_context=context,
            message=f"escalating_retry_{context.retry_count}",
            should_retry=True,
        )
    
    """
    ## Expected behaviour after these changes

    Query: "What is the role of CNN layers in the Transformer architecture?"

    Attempt 1:
    Answerability check:
        Sees "two convolutions with kernel size 1" describing FFN
        Identifies: CNN mentioned as analogy only, not as actual component
        answerable=false, is_false_premise=true
    → Skip LLM, go to correction agent

    Attempt 2 (escalating retry #1):
    Query: "...explicitly state: does CNN exist as a component or only as analogy?"
    Answerability check: still false (document still only has FFN/convolution analogy)
    → OR if it passes → LLM forced to address premise → evaluator checks premise_addressed
    
    Attempt 3 (escalating retry #2):
    Hard NOT FOUND enforcement
    → LLM says "NOT USED in this document"
    → Evaluator: premise_addressed=true, supported=true
    → task_success with correct answer

    Final answer:
    "The Transformer does not use CNN layers. The paper explicitly states it 
    dispenses with recurrence and convolutions entirely. The position-wise 
    feed-forward networks are sometimes described as mathematically similar 
    to two convolutions with kernel size 1, but these are standard linear 
    transformations — not actual CNN layers."
    """

    def _recover_low_confidence(self, context: AgentContext) -> AgentResult:
        """Reformulate with more specificity for low confidence answers."""
        context.retry_count += 1
        suffixes = [
            " Be specific and cite exact sections.",
            " Focus only on what the document explicitly states.",
            " Summarise the most directly relevant points.",
        ]
        context.query = (
            context.query.rstrip(".")
            + suffixes[context.retry_count % len(suffixes)]
        )
        context.errors = []
        context.llm_raw_output = ""
        context.validated_response = None
        logger.info(
            "Low confidence retry #%d — run='%s'",
            context.retry_count, context.run_id,
        )
        return AgentResult(
            success=True,
            updated_context=context,
            message=f"low_confidence_retry_{context.retry_count}",
            should_retry=True,
        )

    def _recover_retrieval_failure(self, context: AgentContext) -> AgentResult:
        """Simplify the query for retrieval failure."""
        context.retry_count += 1
        words = context.query.split()
        if len(words) > 8:
            context.query = " ".join(words[:6]) + "?"
        else:
            context.query = context.query.rstrip(".") + " Provide a brief answer."
        context.errors = []
        context.llm_raw_output = ""
        context.validated_response = None
        logger.info(
            "Retrieval failure retry #%d — run='%s'",
            context.retry_count, context.run_id,
        )
        return AgentResult(
            success=True,
            updated_context=context,
            message=f"retrieval_retry_{context.retry_count}",
            should_retry=True,
        )

    def _improve(self, context: AgentContext) -> AgentResult:
        """Max retries exhausted or unanswerable — return best possible response."""
        logger.warning(
            "Improve step — run='%s' reason='%s' retries=%d",
            context.run_id, context.retry_reason, context.retry_count,
        )

        if context.retry_reason == "not_answerable":
            # Document does not contain this information — clean NOT FOUND
            context.final_answer = "NOT FOUND in the provided document."
            context.confidence = 0.0
            recovery_strategy = "not_found"

        elif context.retry_reason == "hallucination":
            context.final_answer = "NOT FOUND in the provided document."
            context.confidence = 0.0
            recovery_strategy = "not_found_after_hallucination"

        elif context.retry_reason == "low_confidence":
            context.final_answer = _CLARIFICATION_REQUEST
            context.confidence = 0.0
            recovery_strategy = "clarification_requested"

        else:
            context.final_answer = _SAFE_FALLBACK
            context.confidence = 0.0
            recovery_strategy = "safe_fallback"

        self._log_to_improvement_loop(context, recovery_strategy, False)

        return AgentResult(
            success=False,
            updated_context=context,
            message=f"improve_{recovery_strategy}",
            should_retry=False,
        )

    def _log_to_improvement_loop(
        self,
        context: AgentContext,
        recovery_strategy: str,
        recovery_succeeded: bool,
    ) -> None:
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
                hallucination_type=context.hallucination_type or context.retry_reason,
                unmatched_citations=context.unmatched_citations,
                retry_count=context.retry_count,
                recovery_strategy=recovery_strategy,
                final_answer=context.final_answer,
                recovery_succeeded=recovery_succeeded,
            ))
        except Exception as exc:
            logger.error("Failed to write hallucination log: %s", exc)