"""Retrieval agent — runs hierarchical retrieval and context pruning."""

from app.agents.base import BaseAgent, AgentContext, AgentResult
from app.rag.retriever import retrieve
from app.rag.context_pruner import prune, build_context_string
from app.clients.manager import get as get_client


class RetrievalAgent(BaseAgent):
    """Performs two-stage hierarchical retrieval then prunes the context."""

    name = "retrieval_agent"

    def run(self, context: AgentContext) -> AgentResult:
        client_cfg = get_client(context.client_id)
        coarse_k = client_cfg.coarse_k if client_cfg else None
        fine_k = client_cfg.fine_k if client_cfg else None
        max_tokens = client_cfg.max_context_tokens if client_cfg else None

        try:
            chunks = retrieve(
                query=context.query,
                client_id=context.client_id,
                coarse_k=coarse_k,
                fine_k=fine_k,
            )
        except Exception as exc:
            return self._fail(context, f"retrieval_error: {exc}")

        if not chunks:
            return self._fail(context, "no_chunks_retrieved")

        context.retrieved_chunks = chunks

        pruned = prune(chunks, max_tokens=max_tokens, query=context.query)
        context.pruned_chunks = pruned
        context.context_string = build_context_string(pruned)
        context.sources_used = list({c.source for c in pruned})

        self.logger.debug(
            "Retrieved %d chunks, pruned to %d for client '%s'.",
            len(chunks),
            len(pruned),
            context.client_id,
        )
        return AgentResult(success=True, updated_context=context, message=f"retrieved_{len(pruned)}_chunks")
