"""Base agent class and shared context/result data models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Shared context passed between agents during a query run."""

    query: str
    client_id: str
    run_id: str                               # unique ID per query run
    max_retries: int = 2

    # Populated by agents as the pipeline progresses
    retrieved_chunks: list[Any] = field(default_factory=list)
    pruned_chunks: list[Any] = field(default_factory=list)
    context_string: str = ""
    llm_raw_output: str = ""
    validated_response: Any = None            # RAGResponse | None
    final_answer: str = ""
    sources_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    retry_count: int = 0
    retry_reason: str = ""
    strict_mode: bool = False
    signals: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Populated by ValidationAgent when hallucination is detected
    hallucination_faithfulness_score: float = 0.0
    hallucination_type: str = ""
    unmatched_citations: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Standard result returned by every agent run() call."""

    success: bool
    updated_context: AgentContext
    message: str = ""
    should_retry: bool = False


class BaseAgent(ABC):
    """Abstract base for all agents in the pipeline."""

    name: str = "base_agent"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"agent.{self.name}")

    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        """Execute this agent's logic on *context*.

        Must return an :class:`AgentResult`. Never raise — catch internally
        and return a failed result with the error in context.errors.
        """
        ...

    def _fail(self, context: AgentContext, reason: str) -> AgentResult:
        """Convenience helper for returning a failure result."""
        context.errors.append(f"{self.name}: {reason}")
        self.logger.warning("Agent '%s' failed: %s", self.name, reason)
        return AgentResult(success=False, updated_context=context, message=reason)
