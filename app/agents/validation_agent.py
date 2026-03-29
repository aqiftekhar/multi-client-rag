"""Validation agent — schema validates LLM output and checks source attribution."""

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.validation.schema_validator import validate_rag_response, check_source_attribution
from app.evaluation.signals import record, SignalType


class ValidationAgent(BaseAgent):
    """Validates the LLM's raw output for schema compliance and grounded sources."""

    name = "validation_agent"
    CONFIDENCE_THRESHOLD = 0.4

    def run(self, context: AgentContext) -> AgentResult:
        if not context.llm_raw_output:
            return self._fail(context, "no_llm_output_to_validate")

        result = validate_rag_response(context.llm_raw_output)
        context.validated_response = result.parsed
        context.confidence = result.confidence

        if not result.is_valid:
            record(
                SignalType.VALIDATION_FAILURE,
                client_id=context.client_id,
                query=context.query,
                details={"errors": result.errors, "run_id": context.run_id},
            )
            context.errors.extend(result.errors)
            self.logger.warning(
                "Validation failed for run '%s': %s", context.run_id, result.errors
            )
            # Still continue — we have a best-effort parsed response

        # Source attribution check
        if result.parsed:
            retrieved_doc_ids = [c.doc_id for c in context.retrieved_chunks]
            hallucinated = check_source_attribution(result.parsed, retrieved_doc_ids)
            if hallucinated:
                record(
                    SignalType.HALLUCINATION_DETECTED,
                    client_id=context.client_id,
                    query=context.query,
                    details={"hallucinated_sources": hallucinated, "run_id": context.run_id},
                )
                context.errors.append(f"hallucinated_sources: {hallucinated}")

        # Low confidence check
        if context.confidence < self.CONFIDENCE_THRESHOLD:
            record(
                SignalType.LOW_CONFIDENCE,
                client_id=context.client_id,
                query=context.query,
                details={"confidence": context.confidence, "run_id": context.run_id},
            )
            return AgentResult(
                success=False,
                updated_context=context,
                message=f"low_confidence: {context.confidence:.2f}",
                should_retry=True,
            )

        if result.parsed:
            context.final_answer = result.parsed.answer

        return AgentResult(success=True, updated_context=context, message="validation_passed")
