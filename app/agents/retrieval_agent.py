"""Retrieval agent — hierarchical retrieval with HyDE and wider search on hallucination retry."""

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.rag.retriever import retrieve
from app.rag.context_pruner import prune, build_context_string
from app.clients.manager import get as get_client
from app.config import get_settings


class RetrievalAgent(BaseAgent):
    """Two-stage retrieval with HyDE and automatic expansion on hallucination recovery.

    Normal run:
        coarse_k = config default
        HyDE disabled (fast path for simple factual queries)

    On hallucination retry:
        coarse_k doubled → more candidates for cross-encoder
        HyDE enabled → better embedding for semantic search

    HyDE is only activated on retry because:
        1. It adds an extra LLM call (~5-10s)
        2. Most simple queries don't need it
        3. It is most valuable when initial retrieval returned wrong chunks
    """

    name = "retrieval_agent"

    def run(self, context: AgentContext) -> AgentResult:
        cfg = get_settings()
        client_cfg = get_client(context.client_id)
        coarse_k = client_cfg.coarse_k if client_cfg else None
        fine_k = client_cfg.fine_k if client_cfg else None
        max_tokens = client_cfg.max_context_tokens if client_cfg else None

        # On hallucination retry: expand search + activate HyDE
        use_hyde = False
        if context.strict_mode and context.retry_reason == "hallucination":
            base_k = coarse_k or cfg.coarse_k
            coarse_k = base_k * 2
            use_hyde = True
            self.logger.info(
                "Hallucination recovery: coarse_k=%d HyDE=True run='%s'",
                coarse_k, context.run_id,
            )

        try:
            chunks = retrieve(
                query=context.query,
                client_id=context.client_id,
                coarse_k=coarse_k,
                fine_k=fine_k,
                use_hyde=use_hyde,
                ollama_host=cfg.ollama_host if use_hyde else None,
                ollama_model=cfg.ollama_model if use_hyde else None,
            )
        except Exception as exc:
            return self._fail(context, f"retrieval_error: {exc}")

        if not chunks:
            context.retry_reason = "retrieval_failure"
            return self._fail(context, "no_chunks_retrieved")

        context.retrieved_chunks = chunks
        pruned = prune(chunks, max_tokens=max_tokens, query=context.query)
        context.pruned_chunks = pruned
        context.context_string = build_context_string(pruned)
        context.sources_used = list({c.source for c in pruned})

        self.logger.debug(
            "Retrieved %d chunks, pruned to %d (client=%s, hyde=%s)",
            len(chunks), len(pruned), context.client_id, use_hyde,
        )
        return AgentResult(
            success=True,
            updated_context=context,
            message=f"retrieved_{len(pruned)}_chunks",
        )