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
You follow a strict four-step process before answering.

STEP 1 — ASSUMPTION CHECK
Identify any assumptions embedded in the question.
Examples:
  - "What is the role of CNN layers in Transformer?" assumes CNN layers exist in Transformer
  - "Why does X use method Y?" assumes X uses method Y
For each assumption, check: does the document confirm or deny this assumption?
If the document denies an assumption, your answer MUST start by correcting it.

STEP 2 — EXTRACT (Quote-first)
Find the exact sentences from the context relevant to the question.
Copy them word for word. Do not paraphrase yet.
If no relevant sentences exist → answer is NOT FOUND.

STEP 3 — CONTRADICTION CHECK
Does the document explicitly say the opposite of what the question assumes?
Critical rules:
  - "can be seen as X" or "analogous to X" does NOT mean "is X" or "uses X"
  - "dispensing with X" or "without X" means X is explicitly NOT used
  - If document removes or replaces something, say so directly
  ANALOGY ≠ IMPLEMENTATION. DESCRIPTION ≠ EXISTENCE.

STEP 4 — ANSWER
If Step 1 found a false assumption → correct it first, then explain what actually exists
If Step 3 found a contradiction → lead with the contradiction
If no relevant content exists → say NOT FOUND
Never answer a false premise as if it were true

Respond ONLY with valid JSON:
{
  "answer": "<your answer>",
  "sources": ["<source_filename>"],
  "confidence": <0.0-1.0>,
  "needs_clarification": <true|false>,
  "extracted_quotes": ["<exact quote 1>", "<exact quote 2>"],
  "is_inferred": <true|false>,
  "false_premise_detected": <true|false>,
  "false_premise_explanation": "<what the question assumed wrongly, or empty string>"
}

Rules:
- false_premise_detected: true if the question assumed something the document contradicts
- If false_premise_detected is true, confidence should be 0.8+ because you are correcting clearly
- If is_inferred is true, confidence must be below 0.5
- Return ONLY the JSON — no markdown, no preamble, no extra text"""


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
                    self._model,
                    models,
                    self._model,
                )
            else:
                logger.info("Ollama ready. Model '%s' found.", self._model)
        except Exception as exc:
            logger.warning(
                "Cannot reach Ollama at %s: %s — run: ollama serve",
                self._ollama_host,
                exc,
            )

    def run(self, query: str, client_id: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())[:12]
        context = AgentContext(query=query, client_id=client_id, run_id=run_id)
        logger.info(
            "Pipeline start | run_id=%s | client=%s | query=%.60s",
            run_id,
            client_id,
            query,
        )

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
                # 4. Evaluator — catches distorted grounding and false premises
                eval_result = self._evaluate_answer(context)

                # Any of these means the answer is not reliable
                should_retry_eval = (
                    not eval_result.get("supported", True)
                    or eval_result.get("false_premise_missed", False)
                    or eval_result.get("analogy_converted_to_fact", False)
                    or eval_result.get("contradiction_ignored", False)
                    or eval_result.get("scope_drift", False)
                )

                if should_retry_eval:
                    # Evaluator says answer is not directly supported or ignores a contradiction
                    logger.warning(
                        "Evaluator flagged answer as unsupported — run='%s' reason: %s",
                        context.run_id, eval_result.get("reasoning", "")
                    )
                    context.retry_reason = "hallucination"
                    context.strict_mode = True
                    context.hallucination_type = "grounded_but_distorted"
                    context.confidence = max(0.0, context.confidence + eval_result.get("confidence_adjustment", -0.3))
                    context.final_answer = ""
                    context.llm_raw_output = ""
                    correction_result = self.correction_agent.run(context)
                    context = correction_result.updated_context
                    if not correction_result.should_retry:
                        break
                    continue

                # Apply confidence adjustment even on pass
                adjustment = eval_result.get("confidence_adjustment", 0.0)
                if adjustment < 0:
                    context.confidence = max(0.1, context.confidence + adjustment)
                    logger.info(
                        "Evaluator reduced confidence by %.2f — run='%s' (is_inferred=%s)",
                        abs(adjustment), context.run_id, eval_result.get("is_inferred")
                    )

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

        logger.info(
            "Pipeline end | run_id=%s | confidence=%.2f | retries=%d",
            run_id,
            context.confidence,
            context.retry_count,
        )

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
        user_message = (
            f"Context:\n{context.context_string}\n\nQuestion: {context.query}"
        )

        # regenerate_answer(strict_mode=True) — injected after hallucination
        if context.strict_mode:
            user_message += (
                "\n\nSTRICT MODE: A previous response was flagged as not grounded "
                "in the provided context. You MUST only use information explicitly "
                "present in the context above. If the context does not contain a "
                "clear answer, respond with confidence below 0.4 and "
                "needs_clarification set to true. Do NOT infer or use outside knowledge."
            )
            logger.debug("strict_mode injected for run '%s'", context.run_id)

        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "options": {
                "temperature": 0.1,  # low temp for factual answers
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
    
    def _evaluate_answer(self, context: AgentContext) -> dict:
        """Evaluator — second LLM call to judge grounding and premise accuracy.

        Catches:
        - grounded but distorted answers (analogy → fact)
        - false premise questions that were answered as if premise is true
        - answers that ignored explicit contradictions in the document
        """
        if not context.llm_raw_output or not context.context_string:
            return {
                "supported": True,
                "is_inferred": False,
                "premise_challenged": True,
                "reasoning": "",
                "confidence_adjustment": 0.0,
            }

        evaluator_prompt = f"""You are a strict fact-checker for a RAG system.

    Document context:
    {context.context_string}

    Original question:
    {context.query}

    Answer given:
    {context.final_answer or context.llm_raw_output}

    Evaluate the answer against these four criteria:

    CRITERION 1 — FALSE PREMISE
    Did the question contain a false assumption?
    Did the answer CORRECT that false assumption, or did it answer as if the assumption is true?
    Example of FAILURE: question asks about CNN in Transformer, answer describes FFN without saying CNN doesn't exist
    Example of PASS: question asks about CNN in Transformer, answer says "Transformer does not use CNN layers"

    CRITERION 2 — ANALOGY VS FACT
    Did the answer convert an analogy into a factual claim?
    Example of FAILURE: document says "can be seen as convolution", answer says "uses convolution"
    Example of PASS: answer says "the FFN is described as mathematically similar to convolution but is not a CNN"

    CRITERION 3 — CONTRADICTION IGNORED
    Did the document explicitly deny something the answer asserted or implied?
    Example of FAILURE: document says "dispensing with convolutions entirely", answer talks about convolutions as if they exist

    CRITERION 4 — SCOPE DRIFT
    Did the answer drift to a related topic instead of addressing the actual question?
    Example of FAILURE: question about CNN layers, answer describes FFN without connecting back to CNN

    Respond ONLY with this JSON:
    {{
    "supported": <true if answer directly and accurately addresses the question>,
    "false_premise_addressed": <true if the answer correctly challenged a false assumption>,
    "false_premise_missed": <true if there was a false premise but answer ignored it>,
    "analogy_converted_to_fact": <true if answer treated analogy as implementation>,
    "contradiction_ignored": <true if document explicitly denies something answer asserts>,
    "scope_drift": <true if answer addressed a related topic instead of the actual question>,
    "reasoning": "<one clear sentence explaining your verdict>",
    "confidence_adjustment": <float -0.5 to 0.0>
    }}"""

        try:
            import re, json
            payload = {
                "model": self._model,
                "stream": False,
                "messages": [{"role": "user", "content": evaluator_prompt}],
                "options": {"temperature": 0.0, "num_predict": 300},
            }
            response = httpx.post(
                f"{self._ollama_host}/api/chat",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            raw = response.json()["message"]["content"].strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                logger.info(
                    "Evaluator — run='%s' supported=%s false_premise_missed=%s "
                    "analogy_converted=%s contradiction_ignored=%s scope_drift=%s | %s",
                    context.run_id,
                    result.get("supported"),
                    result.get("false_premise_missed"),
                    result.get("analogy_converted_to_fact"),
                    result.get("contradiction_ignored"),
                    result.get("scope_drift"),
                    result.get("reasoning", ""),
                )
                return result
        except Exception as exc:
            logger.warning("Evaluator failed (non-fatal): %s", exc)

        return {
            "supported": True,
            "false_premise_missed": False,
            "reasoning": "",
            "confidence_adjustment": 0.0,
        }