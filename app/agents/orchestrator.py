"""Orchestrator — coordinates the multi-agent RAG pipeline using Ollama as LLM."""

import logging
import uuid
from typing import Any

import httpx

from app.agents.base import AgentContext
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.correction_agent import CorrectionAgent
from app.agents.evaluation_agent import EvaluationAgent
from app.config import get_settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer ONLY based on the provided context. Never invent facts.
For every claim, reference the source document using [N] notation matching the context.
Respond ONLY with a valid JSON object in this exact schema:
{
  "answer": "<your answer here>",
  "sources": ["<doc_id_1>", "<doc_id_2>"],
  "confidence": <float 0.0-1.0>,
  "needs_clarification": <true|false>
}
confidence should reflect how well the context supports your answer.
If context is insufficient, set confidence below 0.4 and needs_clarification to true.
Return ONLY the JSON — no markdown, no explanation, no extra text."""


class Orchestrator:
    def __init__(self) -> None:
        self.retrieval_agent = RetrievalAgent()
        self.validation_agent = ValidationAgent()
        self.correction_agent = CorrectionAgent()
        self.evaluation_agent = EvaluationAgent()
        cfg = get_settings()
        self._ollama_host = cfg.ollama_host.rstrip("/")
        self._model = cfg.ollama_model
        self._check_ollama()

    def _check_ollama(self) -> None:
        """Warn on startup if Ollama is unreachable or model is missing."""
        try:
            r = httpx.get(f"{self._ollama_host}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            model_base = self._model.split(":")[0]
            if not any(model_base in m for m in models):
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s — run: ollama pull %s",
                    self._model, models, self._model,
                )
            else:
                logger.info("Ollama ready. Model '%s' found.", self._model)
        except Exception as exc:
            logger.warning("Cannot reach Ollama at %s: %s — run: ollama serve", self._ollama_host, exc)

    def run(self, query: str, client_id: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())[:12]
        context = AgentContext(query=query, client_id=client_id, run_id=run_id)
        logger.info("Pipeline start | run_id=%s | client=%s | query=%.60s", run_id, client_id, query)

        while True:
            # 1. Retrieve
            retrieval_result = self.retrieval_agent.run(context)
            context = retrieval_result.updated_context
            if not retrieval_result.success:
                correction_result = self.correction_agent.run(context)
                context = correction_result.updated_context
                if not correction_result.should_retry:
                    break
                continue

            # 2. LLM call (Ollama)
            try:
                context = self._call_llm(context)
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                context.errors.append(f"llm_error: {exc}")
                correction_result = self.correction_agent.run(context)
                context = correction_result.updated_context
                if not correction_result.should_retry:
                    break
                continue

            # 3. Validate
            validation_result = self.validation_agent.run(context)
            context = validation_result.updated_context
            if validation_result.success:
                break
            if validation_result.should_retry:
                correction_result = self.correction_agent.run(context)
                context = correction_result.updated_context
                if not correction_result.should_retry:
                    break
                continue
            break

        # 4. Evaluate (always runs)
        self.evaluation_agent.run(context)

        logger.info("Pipeline end | run_id=%s | confidence=%.2f | retries=%d",
                    run_id, context.confidence, context.retry_count)

        return {
            "run_id": run_id,
            "query": query,
            "answer": context.final_answer or "Unable to generate a reliable answer.",
            "confidence": context.confidence,
            "sources": context.sources_used,
            "chunks_used": len(context.pruned_chunks),
            "retry_count": context.retry_count,
            "errors": context.errors,
        }

    def _call_llm(self, context: AgentContext) -> AgentContext:
        """Call Ollama /api/chat and store the response in context."""
        user_message = f"Context:\n{context.context_string}\n\nQuestion: {context.query}"

        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "options": {
                "temperature": 0.1,   # low temp for factual answers
                "num_predict": 1024,
            },
        }

        response = httpx.post(
            f"{self._ollama_host}/api/chat",
            json=payload,
            timeout=120.0,  # local models on CPU can be slow
        )
        response.raise_for_status()

        context.llm_raw_output = response.json()["message"]["content"]
        logger.debug("Ollama response: %.120s", context.llm_raw_output)
        return context


