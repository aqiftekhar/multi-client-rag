"""Hierarchical retrieval: coarse semantic search → MMR fine reranking."""

import logging
from dataclasses import dataclass

import numpy as np

from app.config import get_settings
from app.db.chroma_client import get_or_create_collection
from app.rag.embedder import embed_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its metadata and relevance score."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    chunk_index: int
    score: float          # cosine similarity (0–1)
    metadata: dict


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mmr(
    query_vec: list[float],
    candidates: list[RetrievedChunk],
    candidate_embeddings: list[list[float]],
    k: int,
    lambda_: float = 0.5,
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance reranking.

    Balances relevance to the query and diversity among selected chunks.
    λ=1 → pure relevance, λ=0 → pure diversity.
    """
    if not candidates:
        return []

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < k and remaining:
        mmr_scores: list[tuple[int, float]] = []
        for idx in remaining:
            relevance = _cosine_similarity(query_vec, candidate_embeddings[idx])
            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(
                    _cosine_similarity(candidate_embeddings[idx], candidate_embeddings[sel])
                    for sel in selected_indices
                )
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            mmr_scores.append((idx, mmr_score))

        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def retrieve(
    query: str,
    client_id: str,
    coarse_k: int | None = None,
    fine_k: int | None = None,
) -> list[RetrievedChunk]:
    """Two-stage hierarchical retrieval for *query* against *client_id*'s corpus.

    Stage 1 — Coarse: broad semantic search returning top *coarse_k* candidates.
    Stage 2 — Fine: MMR reranking to top *fine_k* diverse, relevant chunks.
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

    # Stage 1: Coarse retrieval
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=actual_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    raw_embeddings = results["embeddings"][0]

    candidates: list[RetrievedChunk] = []
    candidate_vecs: list[list[float]] = []

    for cid, doc, meta, dist, emb in zip(ids, documents, metadatas, distances, raw_embeddings):
        similarity = 1.0 - dist  # ChromaDB cosine distance → similarity
        candidates.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                source=meta.get("source", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=similarity,
                metadata=meta,
            )
        )
        candidate_vecs.append(list(emb))

    logger.debug(
        "Stage-1 retrieved %d candidates for query '%s...' (client=%s).",
        len(candidates),
        query[:40],
        client_id,
    )

    # Stage 2: MMR fine reranking
    final_k = min(fine_k, len(candidates))
    reranked = _mmr(query_vec, candidates, candidate_vecs, k=final_k)

    logger.debug("Stage-2 MMR selected %d chunks.", len(reranked))
    return reranked
