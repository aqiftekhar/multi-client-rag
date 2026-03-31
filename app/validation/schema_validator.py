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
    sources: list[str] = []
    confidence: float = 0.5
    needs_clarification: bool = False
    extracted_quotes: list[str] = []
    is_inferred: bool = False
    false_premise_detected: bool = False
    false_premise_explanation: str = ""

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

        # If model itself flagged the answer as inferred, reduce confidence
        # Model flagged its own answer as inferred — reduce confidence
    if parsed.is_inferred and parsed.confidence > 0.5:
        logger.warning(
            "Model flagged answer as inferred — reducing confidence %.2f → %.2f",
            parsed.confidence, 0.45,
        )
        parsed.confidence = min(parsed.confidence, 0.45)

    # Model detected a false premise — log it clearly
    if parsed.false_premise_detected:
        logger.info(
            "False premise detected — explanation: %s",
            parsed.false_premise_explanation,
        )

    logger.debug(
        "Validation passed. Confidence=%.2f is_inferred=%s quotes=%d",
        parsed.confidence, parsed.is_inferred, len(parsed.extracted_quotes)
    )
    return ValidationResult(
        is_valid=True,
        parsed=parsed,
        confidence=parsed.confidence,
    )


def check_source_attribution(
    response: RAGResponse,
    retrieved_chunks: list,
) -> list[str]:
    """Check cited sources against what was actually retrieved.

    Matches against filenames and doc_id prefixes — not raw UUIDs
    which models cannot reliably reproduce verbatim.

    Returns list of citations that don't match anything retrieved.
    """
    if not response.sources:
        return []

    valid_references: set[str] = set()
    for chunk in retrieved_chunks:
        valid_references.add(chunk.source.lower())
        valid_references.add(chunk.doc_id.lower())
        valid_references.add(chunk.doc_id[:8].lower())
        source_base = chunk.source.lower().replace(".pdf", "").replace(".txt", "")
        valid_references.add(source_base)

    hallucinated = []
    for cited in response.sources:
        cited_lower = cited.lower().strip()
        matched = any(
            cited_lower in ref or ref in cited_lower
            for ref in valid_references
        )
        if not matched:
            hallucinated.append(cited)

    if hallucinated:
        logger.warning("Unmatched source citations: %s", hallucinated)

    return hallucinated


"""
## What changes and why

| Problem | Fix |
|---|---|
| Model converted analogy to fact ("can be seen as" → "is") | Quote-first step forces extraction before generation |
| Model ignored "dispensing with convolutions entirely" | Contradiction check step — explicit denials take priority |
| Cross-encoder passed it because text was topically related | Evaluator LLM call checks semantic accuracy, not just topic similarity |
| Model was overconfident | `is_inferred: true` in JSON response → confidence automatically reduced below 0.5 |
| Distorted grounding not caught | Evaluator `contradiction_ignored: true` triggers full hallucination recovery |

---

## What your system now catches

| Hallucination type | Caught by |
|---|---|
| Answer unrelated to context | Cross-encoder faithfulness score |
| Citation of non-existent sources | Source attribution check |
| Analogy converted to fact | Evaluator `is_inferred` + `supported` check |
| Explicit contradiction ignored | Evaluator `contradiction_ignored` flag |
| Model self-aware uncertainty | `is_inferred: true` in JSON → confidence reduction |

---

## Test it with your exact case

After applying these changes, ask again:
```
What is the role of CNN layers in the Transformer architecture?
"""