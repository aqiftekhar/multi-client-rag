"""Validation agent — contain and verify steps of the hallucination strategy."""

import logging

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.validation.schema_validator import validate_rag_response, check_source_attribution
from app.evaluation.signals import record, SignalType

logger = logging.getLogger(__name__)

FAITHFULNESS_THRESHOLD = -2.0
CONFIDENCE_THRESHOLD = 0.4


def _faithfulness_score(answer: str, chunks: list) -> float:
    """Score answer grounding using the cross-encoder we already have.

    Scores (answer, chunk_text) pairs and returns the MAX score.
    Score interpretation:
      > 2.0   strongly grounded
      0 to 2  moderately grounded
      -2 to 0 weak grounding
      < -2    not grounded = hallucination
    """
    if not chunks or not answer.strip():
        return 0.0
    try:
        from app.rag.retriever import _get_cross_encoder
        cross_encoder = _get_cross_encoder()
        pairs = [
            (answer, chunk.metadata.get("raw_text", chunk.text))
            for chunk in chunks
        ]
        scores = cross_encoder.predict(pairs)
        max_score = float(max(scores))
        logger.debug(
            "Faithfulness scores: %s → max=%.4f",
            [f"{s:.2f}" for s in scores], max_score
        )
        return max_score
    except Exception as exc:
        logger.warning("Faithfulness check failed (non-fatal): %s", exc)
        return 1.0  # fail open


class ValidationAgent(BaseAgent):
    name = "validation_agent"

    def run(self, context: AgentContext) -> AgentResult:
        if not context.llm_raw_output:
            return self._fail(context, "no_llm_output_to_validate")

        # ── Schema validation ─────────────────────────────────────────────────
        result = validate_rag_response(context.llm_raw_output)
        context.validated_response = result.parsed
        context.confidence = result.confidence

        if not result.is_valid:
            record(SignalType.VALIDATION_FAILURE, client_id=context.client_id,
                   query=context.query,
                   details={"errors": result.errors, "run_id": context.run_id})
            context.errors.extend(result.errors)

        if not result.parsed:
            return self._fail(context, "could_not_parse_llm_output")

        # ── Source attribution ────────────────────────────────────────────────
        unmatched = check_source_attribution(result.parsed, context.retrieved_chunks)
        if unmatched:
            context.unmatched_citations = unmatched
            context.confidence = max(0.1, context.confidence - 0.15)
            record(SignalType.HALLUCINATION_DETECTED, client_id=context.client_id,
                   query=context.query,
                   details={"type": "citation_mismatch", "unmatched": unmatched,
                            "run_id": context.run_id})
            logger.warning("Citation mismatch run '%s': %s", context.run_id, unmatched)

        # ── VERIFY — faithfulness check ───────────────────────────────────────
        faithfulness = _faithfulness_score(result.parsed.answer, context.retrieved_chunks)
        context.hallucination_faithfulness_score = faithfulness

        if faithfulness < FAITHFULNESS_THRESHOLD:
            # ── CONTAIN ───────────────────────────────────────────────────────
            context.hallucination_type = "content_not_grounded"
            context.final_answer = ""
            context.llm_raw_output = ""
            context.validated_response = None
            context.confidence = 0.0
            context.retry_reason = "hallucination"
            context.strict_mode = True

            record(SignalType.HALLUCINATION_DETECTED, client_id=context.client_id,
                   query=context.query,
                   details={"type": "content_not_grounded",
                            "faithfulness_score": round(faithfulness, 4),
                            "threshold": FAITHFULNESS_THRESHOLD,
                            "run_id": context.run_id})

            logger.warning(
                "HALLUCINATION CONTAINED — run='%s' faithfulness=%.4f. "
                "Passing to recovery.", context.run_id, faithfulness
            )
            return AgentResult(
                success=False, updated_context=context,
                message=f"hallucination_contained|score={faithfulness:.4f}",
                should_retry=True,
            )

        # ── Confidence gate ───────────────────────────────────────────────────
        if context.confidence < CONFIDENCE_THRESHOLD:
            context.retry_reason = "low_confidence"
            record(SignalType.LOW_CONFIDENCE, client_id=context.client_id,
                   query=context.query,
                   details={"confidence": context.confidence, "run_id": context.run_id})
            return AgentResult(
                success=False, updated_context=context,
                message=f"low_confidence={context.confidence:.2f}",
                should_retry=True,
            )

        context.final_answer = result.parsed.answer
        logger.debug("Validation passed — run='%s' confidence=%.2f faithfulness=%.4f",
                     context.run_id, context.confidence, faithfulness)
        return AgentResult(success=True, updated_context=context,
                           message="validation_passed")