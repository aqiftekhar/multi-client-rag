"""Structured output validation for LLM responses using Pydantic."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Expected structure of an LLM RAG answer."""

    answer: str
    sources: list[str]           # List of source identifiers cited
    confidence: float            # 0.0–1.0
    needs_clarification: bool = False


@dataclass
class ValidationResult:
    """Result of validating an LLM output."""

    is_valid: bool
    parsed: RAGResponse | None
    errors: list[str] = field(default_factory=list)
    confidence: float = 0.0


def _extract_json_block(text: str) -> str:
    """Try to extract a JSON block from LLM output."""
    # Try ```json ... ``` first
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try bare { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def validate_rag_response(raw_output: str) -> ValidationResult:
    """Parse and validate the LLM's raw output string into a :class:`RAGResponse`.

    The LLM is prompted to return JSON; this validator attempts to parse it
    and falls back gracefully, returning a low-confidence result.
    """
    import json

    errors: list[str] = []

    json_text = _extract_json_block(raw_output)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s | raw: %.120s", exc, raw_output)
        # Fallback: treat entire output as the answer
        return ValidationResult(
            is_valid=False,
            parsed=RAGResponse(
                answer=raw_output.strip(),
                sources=[],
                confidence=0.2,
                needs_clarification=True,
            ),
            errors=[f"json_parse_error: {exc}"],
            confidence=0.2,
        )

    try:
        parsed = RAGResponse(**data)
    except (ValidationError, TypeError) as exc:
        errors.append(f"schema_validation_error: {exc}")
        logger.warning("Schema validation failed: %s", exc)
        # Best-effort extraction
        answer = data.get("answer", raw_output)
        confidence = float(data.get("confidence", 0.3))
        return ValidationResult(
            is_valid=False,
            parsed=RAGResponse(
                answer=answer,
                sources=data.get("sources", []),
                confidence=confidence,
                needs_clarification=True,
            ),
            errors=errors,
            confidence=confidence,
        )

    logger.debug("Validation passed. Confidence=%.2f", parsed.confidence)
    return ValidationResult(
        is_valid=True,
        parsed=parsed,
        confidence=parsed.confidence,
    )


def check_source_attribution(
    response: RAGResponse,
    retrieved_sources: list[str],
) -> list[str]:
    """Verify that cited sources in *response* are present in *retrieved_sources*.

    Returns a list of hallucinated source IDs (cited but not retrieved).
    """
    hallucinated = [s for s in response.sources if s not in retrieved_sources]
    if hallucinated:
        logger.warning("Hallucinated sources detected: %s", hallucinated)
    return hallucinated
