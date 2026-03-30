"""Hierarchical retrieval: coarse semantic search → cross-encoder reranking.

Stage 1 — Coarse:  broad dense vector search, top coarse_k candidates.
Stage 2 — Rerank:  cross-encoder scores every (query, chunk) pair together
                   and selects top fine_k by true relevance.

Why cross-encoder over MMR:
  MMR optimises for diversity using cosine similarity.
  The cross-encoder reads query + chunk together in one pass —
  it understands relevance the way a human reader would.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.db.chroma_client import get_or_create_collection
from app.rag.embedder import embed_query

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache()
def _get_cross_encoder() -> CrossEncoder:
    """Load and cache the cross-encoder (loaded once on first query)."""
    logger.info("Loading cross-encoder '%s'...", _CROSS_ENCODER_MODEL)
    model = CrossEncoder(_CROSS_ENCODER_MODEL, max_length=512)
    logger.info("Cross-encoder loaded.")
    return model


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its metadata and relevance score."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    chunk_index: int
    score: float          # cross-encoder score after reranking
    metadata: dict


def _rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Rerank candidates using the cross-encoder.

    Scores every (query, raw_chunk_text) pair. Returns top_k
    sorted by score descending.
    """
    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()

    # Use raw_text from metadata where available so the header
    # text doesn't interfere with relevance scoring
    pairs = [
        (query, chunk.metadata.get("raw_text", chunk.text))
        for chunk in candidates
    ]

    scores = cross_encoder.predict(pairs)

    scored = sorted(
        zip(candidates, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    result = []
    for chunk, score in scored[:min(top_k, len(scored))]:
        chunk.score = float(score)
        result.append(chunk)

    logger.debug(
        "Cross-encoder: %d → %d chunks. Top score: %.4f",
        len(candidates),
        len(result),
        result[0].score if result else 0.0,
    )
    return result


def retrieve(
    query: str,
    client_id: str,
    coarse_k: int | None = None,
    fine_k: int | None = None,
) -> list[RetrievedChunk]:
    """Two-stage retrieval for query against client_id's corpus.

    Stage 1 — Coarse dense retrieval (top coarse_k by embedding similarity).
    Stage 2 — Cross-encoder reranking (top fine_k by true relevance).
    """
    cfg = get_settings()
    coarse_k = coarse_k or cfg.coarse_k
    fine_k = fine_k or cfg.fine_k

    query_vec = embed_query(query)
    collection = get_or_create_collection(client_id)

    n_docs = collection.count()
    if n_docs == 0:
        logger.warning("Collection for client '%s' is empty.", client_id)
        return []

    actual_k = min(coarse_k, n_docs)

    # Stage 1: Coarse dense retrieval
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        candidates.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                source=meta.get("source", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=1.0 - dist,
                metadata=meta,
            )
        )

    logger.debug(
        "Stage-1: %d candidates for '%.40s' (client=%s)",
        len(candidates), query, client_id,
    )

    # Stage 2: Cross-encoder reranking
    return _rerank(query, candidates, top_k=fine_k)