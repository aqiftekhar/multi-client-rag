"""Dynamic context pruner — trims retrieved chunks to fit LLM token budget."""

import logging

import tiktoken

from app.config import get_settings
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(_enc.encode(text))


def prune(
    chunks: list[RetrievedChunk],
    max_tokens: int | None = None,
    query: str = "",
) -> list[RetrievedChunk]:
    """Trim *chunks* so total token count ≤ *max_tokens*.

    Chunks are already ranked by relevance (highest first from retriever).
    We greedily include chunks in relevance order until the budget is exhausted.

    Also accounts for:
    - A fixed system-prompt overhead (300 tokens reserved)
    - The query itself
    """
    cfg = get_settings()
    budget = (max_tokens or cfg.max_context_tokens) - 300  # system prompt overhead
    budget -= _tokens(query)

    selected: list[RetrievedChunk] = []
    used = 0

    for chunk in chunks:
        ct = _tokens(chunk.text)
        if used + ct <= budget:
            selected.append(chunk)
            used += ct
        else:
            # Try to fit a truncated version if the chunk is large
            remaining = budget - used
            if remaining > 100:  # only worth including if >100 tokens remain
                encoded = _enc.encode(chunk.text)[:remaining]
                truncated_text = _enc.decode(encoded)
                selected.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=truncated_text + " [truncated]",
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                        score=chunk.score,
                        metadata=chunk.metadata,
                    )
                )
                used += remaining
            break

    logger.debug(
        "Context pruner: %d/%d chunks kept, %d tokens used of %d budget.",
        len(selected),
        len(chunks),
        used,
        budget,
    )
    return selected


def build_context_string(chunks: list[RetrievedChunk]) -> str:
    """Format pruned chunks into an LLM-ready context block with source attribution."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk.source} | Doc: {chunk.doc_id} | Chunk: {chunk.chunk_index}\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)
