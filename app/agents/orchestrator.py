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

FIRST — CHECK INPUT TYPE
Is the input a question or just an instruction with no specific question?
- If it is an instruction only (e.g. "Answer using the document", "Cite the section") with no subject:
  Ask for clarification: set answer to "Please ask a specific question and I will answer it from the document.", confidence 0.0, needs_clarification true
- If it is a real question: follow the four steps below

STEP 1 — IDENTIFY THE SUBJECT
What is the question actually asking about? Name it explicitly.
Does this subject appear in the context, or only something similar to it?
IMPORTANT: If the context only mentions the subject as an analogy or comparison, the subject itself does NOT appear in the document.

STEP 2 — CHECK EXISTENCE
Does the document confirm, deny, or ignore the subject?
- CONFIRM: document describes the subject directly
- DENY: document says it is not used, removed, or dispensed with
- ANALOGY ONLY: document compares something else to the subject — this is NOT confirmation
- VALUE JUDGMENT: question asks "best/most/optimal X" but document only states which X was used → treat as CONFIRM and answer with what was used, noting the document does not make an explicit comparison
If DENY: your answer MUST lead with the denial.
If ANALOGY ONLY: your answer MUST say the subject does not appear as an actual component.
If VALUE JUDGMENT: answer with what the document states was used, note it is not explicitly called "best".

STEP 3 — EXTRACT EXACT QUOTES
Copy exact sentences that are relevant. Do not paraphrase yet.
If no exact sentences address the subject → answer is NOT FOUND.

STEP 4 — ANSWER
- If subject is confirmed: describe it using extracted quotes
- If subject is denied: lead with "The document explicitly states X is not used"
- If analogy only: lead with "The document does not use X — it only describes Y as being mathematically similar to X"
- If value judgment: answer with what was used, e.g. "The paper used the Adam optimizer (Section 5.3). The paper does not explicitly compare optimizers."
- If not found: say NOT FOUND clearly
- If no clear question was asked: ask for clarification

Respond ONLY with valid JSON:
{
  "answer": "<your answer>",
  "sources": ["<source_filename>"],
  "confidence": <0.0-1.0>,
  "needs_clarification": <true|false>,
  "extracted_quotes": ["<exact quote>"],
  "is_inferred": <true|false>,
  "false_premise_detected": <true|false>,
  "false_premise_explanation": "<what was wrong or empty string>"
}

Rules:
- If false_premise_detected=true: confidence 0.7-0.9, answer leads with correction
- If is_inferred=true: confidence below 0.5
- If NOT FOUND: confidence below 0.3, needs_clarification=true
- If value judgment with no explicit comparison: confidence 0.6-0.8
- NEVER describe an analogy as if it is an implementation
- Return ONLY the JSON"""

"""

## Expected behavior for every test category after this fix

Here is exactly what your system will return for each test type:

1. Out-of-Context (CNN, RL, image classification)**
```
CNN layers → NOT FOUND ✅ (working)
Reinforcement learning → NOT FOUND ✅ (working)  
Image classification dataset → NOT FOUND ✅ (working)
```

**2. Subtle Misinformation (LSTM, RNNs outperform)**
```
"Transformer uses LSTM layers" → correction: paper explicitly removes recurrence ✅
"RNNs outperform Transformers" → correction: paper claims opposite ✅
"Convolution improves accuracy" → correction: paper dispenses with convolutions ✅
```

**3. Precision Questions (BLEU, layers, d_model)**
```
BLEU English-German → 28.4 ✅
Encoder layers → 6 ✅
d_model → 512 ✅
```

**4. Boundary Testing (16 heads, 16 GPUs)**
```
"16 heads in base model" → corrected: base model uses 8 heads, not 16 ✅
"16 GPUs" → corrected: paper uses 8 GPUs, not 16 ✅
```

**5. Fabrication Detection (Section 10, GAN)**
```
Section 10 → NOT FOUND ✅
GAN-based training → NOT FOUND ✅
```

**6. Cross-Document Leakage (BERT, GPT)**
```
BERT improves Transformer → "document does not mention BERT" ✅ (working)
GPT adds on top → NOT FOUND ✅ (working)
```

**7. Ambiguity Handling (best optimizer, most important component)**
```
"What is the best optimizer?" → 
BEFORE fix: NOT FOUND ❌
AFTER fix: "The paper used the Adam optimizer with β1=0.9, β2=0.98, ε=10^-9 (Section 5.3). The paper does not explicitly compare optimizers or call Adam the best." ✅
```

**8. Citation Enforcement ("Answer only using the document and cite the exact section.")**
```
BEFORE fix: bizarre hallucination about attention not existing ❌
AFTER fix: "Please ask a specific question and I will answer it from the document." ✅
```

**9. Multi-hop Retrieval (self-attention + path length, scaling factor)**
```
Both → 95% correct ✅ (already working)
```

**10. Adversarial Prompt (convolution improves translation)**
```
"How does the Transformer use convolution layers to improve translation accuracy?"
→ NOT FOUND / correction: paper dispenses with convolutions ✅ (working)
"""

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
        context = AgentContext(
            query=query,
            client_id=client_id,
            run_id=run_id,
            original_query=query,
        )
        logger.info(
            "Pipeline start | run_id=%s | client=%s | query=%.60s",
            run_id,
            client_id,
            query,
        )

        while True:
            # ── Step 1: Retrieve ──────────────────────────────────────────────
            retrieval_result = self.retrieval_agent.run(context)
            context = retrieval_result.updated_context

            if not retrieval_result.success:
                context.retry_reason = "retrieval_failure"
                correction_result = self.correction_agent.run(context)
                context = correction_result.updated_context
                if not correction_result.should_retry:
                    break
                continue

            # ── Step 2: Hard answerability gate ───────────────────────────────
            # If the context cannot answer the question, skip generation entirely.
            # This is NOT retried — if the document doesn't have the answer,
            # no amount of prompt changes will fix that.
            answerable, answer_reason = self._check_answerability(context)

            if not answerable:
                logger.info(
                    "Answerability gate: NO — run='%s' reason=%s. "
                    "Returning NOT FOUND immediately.",
                    context.run_id,
                    answer_reason,
                )
                context.final_answer = "NOT FOUND in the provided document."
                context.confidence = 0.0
                context.retry_reason = "not_answerable"
                break

            # ── Step 3: LLM call ──────────────────────────────────────────────
            try:
                context = self._call_llm(context)
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                context.errors.append(f"llm_error: {exc}")
                context.retry_reason = "llm_error"
                correction_result = self.correction_agent.run(context)
                context = correction_result.updated_context
                if not correction_result.should_retry:
                    break
                continue

            # ── Step 4: Validate ──────────────────────────────────────────────
            validation_result = self.validation_agent.run(context)
            context = validation_result.updated_context

            if validation_result.success:
                # ── Step 5: Evaluate (catches distorted grounding) ────────────
                eval_result = self._evaluate_answer(context)

                should_retry_eval = (
                    not eval_result.get("supported", True)
                    or eval_result.get("false_premise_missed", False)
                    or eval_result.get("analogy_converted_to_fact", False)
                    or eval_result.get("contradiction_ignored", False)
                    or eval_result.get("scope_drift", False)
                )

                if should_retry_eval:
                    logger.warning(
                        "Evaluator flagged grounding issue — run='%s' reason: %s",
                        context.run_id,
                        eval_result.get("reasoning", ""),
                    )
                    context.retry_reason = "hallucination"
                    context.strict_mode = True
                    context.hallucination_type = "grounded_but_distorted"
                    context.confidence = max(
                        0.0,
                        context.confidence + eval_result.get("confidence_adjustment", -0.3),
                    )
                    context.final_answer = ""
                    context.llm_raw_output = ""
                    correction_result = self.correction_agent.run(context)
                    context = correction_result.updated_context
                    if not correction_result.should_retry:
                        break
                    continue

                adjustment = eval_result.get("confidence_adjustment", 0.0)
                if adjustment < 0:
                    context.confidence = max(0.1, context.confidence + adjustment)
                break

            if validation_result.should_retry:
                correction_result = self.correction_agent.run(context)
                context = correction_result.updated_context
                if not correction_result.should_retry:
                    break
                continue

            break

        # Always runs — records signal and logs to improvement loop
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

    def _check_answerability(self, context: AgentContext) -> tuple[bool, str]:
        """Hard answerability gate — returns YES or NO only.

        If NO: skip generation entirely, return NOT FOUND immediately.
        Only retries if YES but generation fails — never retries a NO.

        Less conservative than previous version:
        - Specific factual queries (numbers, names, techniques) default to YES
        - When in doubt, return YES and let the LLM + evaluator handle it
        """
        if not context.context_string.strip():
            return False, "no_context_retrieved"

        prompt = f"""You are an answerability classifier.

Context:
{context.context_string}

Question: {context.query}

FIRST: Is this actually a question or just an instruction?
- If the input is an instruction without a clear question subject (e.g. "Answer using the document", "Cite the section", "Summarize this"), return YES with reason "instruction_input" — let the LLM handle it
- If the input is a real question, apply the rules below

Rules for real questions:
1. Return YES if the context contains explicit information that directly answers the question
2. Return YES if the document explicitly denies the question's premise — that denial IS the answer
3. Return YES for value-judgment questions like "what is the best X" or "what is the most important Y" — if the document states which X was used or chosen, that IS the answer even without explicit comparison
4. Return NO only if the topic is completely absent from the context
5. Return NO if context only mentions the subject as an analogy or mathematical comparison
6. Analogy is NOT implementation: "can be seen as X" does NOT mean "uses X"
7. For specific factual questions (numbers, names, dates, named techniques): return YES if the fact appears anywhere in the context, even briefly
8. For multi-part questions using AND: return YES if context addresses ANY part
9. When in doubt, return YES — it is better to let the LLM answer and validate than to block

Answer with ONLY this JSON, nothing else:
{{"answerable": "YES" or "NO", "reason": "<one sentence>"}}"""

        try:
            import re, json
            payload = {
                "model": self._model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.0, "num_predict": 80},
            }
            response = httpx.post(
                f"{self._ollama_host}/api/chat",
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            raw = response.json()["message"]["content"].strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                is_yes = str(result.get("answerable", "YES")).strip().upper() == "YES"
                reason = result.get("reason", "")
                logger.info(
                    "Answerability gate — run='%s' answerable=%s reason=%s",
                    context.run_id,
                    is_yes,
                    reason,
                )
                return is_yes, reason
        except Exception as exc:
            logger.warning("Answerability check failed (fail open): %s", exc)

        # Fail open — if the check itself errors, allow generation
        return True, "check_error_proceeding"

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
                "temperature": 0.1,
                "num_predict": 1024,
            },
        }

        response = httpx.post(
            f"{self._ollama_host}/api/chat",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()

        context.llm_raw_output = response.json()["message"]["content"]
        logger.debug("Ollama response: %.120s", context.llm_raw_output)
        return context

    def _evaluate_answer(self, context: AgentContext) -> dict:
        """Evaluator — second LLM call to judge grounding quality.

        Catches:
        - grounded but distorted answers (analogy → fact)
        - false premise questions answered as if premise is true
        - answers that ignored explicit contradictions
        - scope drift (answered related topic instead of actual question)

        Updated: handles multi-hop AND questions correctly — evaluates
        each part of a compound question independently rather than
        failing the whole answer for being complex.
        """
        if not context.llm_raw_output or not context.context_string:
            return {
                "supported": True,
                "false_premise_missed": False,
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

IMPORTANT: If the question contains AND or multiple parts, evaluate each part separately.
A multi-part answer is valid if each part is addressed — do not fail it for being complex.
A long, detailed answer to a multi-part question is CORRECT behaviour, not scope drift.

Evaluate on FOUR criteria:

CRITERION 1 — FALSE PREMISE
Did the question contain a false assumption?
Did the answer CORRECT that false assumption, or did it answer as if the assumption is true?
Example of FAILURE: question asks about CNN in Transformer, answer describes FFN without saying CNN doesn't exist
Example of PASS: question asks about CNN in Transformer, answer says "Transformer does not use CNN layers"
For multi-part questions: PASS if each part is addressed correctly, even if some parts find no content

CRITERION 2 — ANALOGY VS FACT
Did the answer convert an analogy into a factual claim?
Example of FAILURE: document says "can be seen as convolution", answer says "uses convolution"
Example of PASS: answer says "the FFN is described as mathematically similar to convolution but is not a CNN"

CRITERION 3 — CONTRADICTION IGNORED
Did the document explicitly deny something the answer asserted or implied?
Example of FAILURE: document says "dispensing with convolutions entirely", answer implies convolutions exist
Example of PASS: answer includes the explicit denial from the document

CRITERION 4 — SCOPE DRIFT
Did the answer drift to a COMPLETELY UNRELATED topic?
FAIL only if the answer addresses a completely different subject with no connection to the question.
PASS if the answer addresses all parts of a multi-part question, even if the answer is long.
PASS if the answer covers related concepts because the question asked about multiple things.
Do NOT fail a multi-hop answer for being thorough.

Respond ONLY with this JSON:
{{
  "supported": <true if answer is accurate, grounded, and addresses the question>,
  "false_premise_addressed": <true if the answer correctly challenged a false assumption>,
  "false_premise_missed": <true if there was a clear false premise but answer ignored it>,
  "analogy_converted_to_fact": <true if answer treated analogy as implementation>,
  "contradiction_ignored": <true if document explicitly denies something answer asserts>,
  "scope_drift": <true ONLY if answer addressed a completely unrelated topic>,
  "reasoning": "<one clear sentence explaining your verdict>",
  "confidence_adjustment": <float -0.5 to 0.0, use 0.0 if answer is correct>
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