"""Query decomposition agent — breaks complex multi-part queries into atomic sub-queries.

When to use:
  Queries containing AND, OR, "both", "compare", "also", "as well as"
  are candidates for decomposition.

How it works:
  1. Detect if query is complex (contains multiple distinct questions)
  2. If yes: split into 2-3 atomic sub-queries
  3. Run retrieval for each sub-query separately
  4. Merge all retrieved chunks (deduped) into a single context
  5. Generate one answer from the merged context

Why this matters:
  "Explain self-attention AND relate it to path length" may retrieve chunks
  about self-attention efficiency but miss the path length section entirely
  if both topics don't co-occur in the same chunk.
  Decomposition retrieves for each independently then merges.
"""

import logging
import re

import httpx

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.config import get_settings

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """Analyze this query and decide if it contains multiple distinct questions.

Query: {query}

Rules:
1. If the query asks about ONE thing (even in detail) → not complex, return original
2. If the query asks about TWO OR MORE distinct concepts → complex, split it
3. Maximum 3 sub-queries
4. Each sub-query must be self-contained and independently answerable
5. Do not split if the parts are tightly coupled (e.g. "why does X cause Y" is one question)

Indicators of complex queries: AND, OR, "both", "compare", "also", "as well as",
"in addition", multiple question marks, "relate X to Y"

Respond ONLY with this JSON:
{{
  "is_complex": true or false,
  "sub_queries": ["sub-query 1", "sub-query 2"]
}}

If not complex, set is_complex=false and sub_queries=["{query}"]"""


class QueryDecompositionAgent(BaseAgent):
    """Detects and decomposes complex multi-part queries."""

    name = "decomposition_agent"

    # Simple heuristic indicators — check before calling LLM
    _COMPLEXITY_INDICATORS = [
        r"\band\b",
        r"\bor\b",
        r"\bboth\b",
        r"\bcompare\b",
        r"\balso\b",
        r"\bas well as\b",
        r"\bin addition\b",
        r"\brelate .+ to\b",
        r"\?.*\?",  # multiple question marks
    ]

    def _looks_complex(self, query: str) -> bool:
        """Quick heuristic check before spending an LLM call."""
        q = query.lower()
        return any(re.search(pat, q) for pat in self._COMPLEXITY_INDICATORS)

    def _decompose(self, query: str, ollama_host: str, model: str) -> list[str]:
        """Call LLM to decompose query. Returns list of sub-queries."""
        try:
            response = httpx.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": _DECOMPOSE_PROMPT.format(query=query),
                        }
                    ],
                    "options": {"temperature": 0.0, "num_predict": 200},
                },
                timeout=20.0,
            )
            response.raise_for_status()
            raw = response.json()["message"]["content"].strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                import json
                result = json.loads(match.group(0))
                if result.get("is_complex") and result.get("sub_queries"):
                    sub_queries = result["sub_queries"]
                    if len(sub_queries) > 1:
                        logger.info(
                            "Query decomposed into %d sub-queries: %s",
                            len(sub_queries),
                            [q[:40] for q in sub_queries],
                        )
                        return sub_queries
        except Exception as exc:
            logger.warning("Query decomposition failed (non-fatal): %s", exc)

        return [query]  # fall back to original

    def run(self, context: AgentContext) -> AgentResult:
        """Detect and decompose complex queries.

        If decomposition happens, retrieves for each sub-query separately
        and merges all chunks into context before generation.
        """
        cfg = get_settings()

        # Quick heuristic check first — skip LLM if clearly not complex
        if not self._looks_complex(context.query):
            logger.debug(
                "Query decomposition skipped — not complex (run='%s')",
                context.run_id,
            )
            return AgentResult(
                success=True,
                updated_context=context,
                message="decomposition_skipped_not_complex",
            )

        sub_queries = self._decompose(
            context.query,
            cfg.ollama_host,
            cfg.ollama_model,
        )

        if len(sub_queries) <= 1:
            return AgentResult(
                success=True,
                updated_context=context,
                message="decomposition_not_needed",
            )

        # Retrieve for each sub-query and merge results
        from app.rag.retriever import retrieve
        from app.rag.context_pruner import build_context_string

        all_chunks = []
        seen_ids = set()

        for sub_q in sub_queries:
            try:
                chunks = retrieve(
                    query=sub_q,
                    client_id=context.client_id,
                )
                for chunk in chunks:
                    if chunk.chunk_id not in seen_ids:
                        all_chunks.append(chunk)
                        seen_ids.add(chunk.chunk_id)
                logger.debug(
                    "Sub-query '%s...' retrieved %d chunks",
                    sub_q[:40], len(chunks),
                )
            except Exception as exc:
                logger.warning("Sub-query retrieval failed: %s", exc)

        if not all_chunks:
            return AgentResult(
                success=True,
                updated_context=context,
                message="decomposition_retrieval_empty",
            )

        # Sort by score descending, keep top fine_k * num_sub_queries
        # from app.config import get_settings
        cfg = get_settings()
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        top_k = cfg.fine_k * len(sub_queries)
        all_chunks = all_chunks[:top_k]

        context.retrieved_chunks = all_chunks

        # Prune to token budget
        from app.rag.context_pruner import prune
        pruned = prune(all_chunks, query=context.query)
        context.pruned_chunks = pruned
        context.context_string = build_context_string(pruned)
        context.sources_used = list({c.source for c in pruned})

        logger.info(
            "Query decomposed: %d sub-queries → %d unique chunks → %d pruned (run='%s')",
            len(sub_queries), len(all_chunks), len(pruned), context.run_id,
        )

        return AgentResult(
            success=True,
            updated_context=context,
            message=f"decomposed_into_{len(sub_queries)}_sub_queries",
        )