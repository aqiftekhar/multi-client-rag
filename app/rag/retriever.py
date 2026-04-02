"""Hybrid retrieval: Dense + BM25 → Reciprocal Rank Fusion → Cross-encoder reranking.

Three-stage pipeline:
  Stage 1A — Dense:   embedding similarity search (ChromaDB), top coarse_k
  Stage 1B — Sparse:  BM25 keyword search, top coarse_k
  Stage 2  — Fusion:  Reciprocal Rank Fusion merges both ranked lists
  Stage 3  — Rerank:  cross-encoder scores every (query, chunk) pair for true relevance

Why hybrid over pure dense:
  Dense search is great for semantic similarity but misses exact terms.
  BM25 is great for exact keywords but misses semantic meaning.
  RRF combines both — a chunk must rank well in at least one list to score high.

Example where hybrid wins over pure dense:
  Query: "dmodel = 512 inner-layer dimensionality"
  Dense alone: may retrieve related architecture chunks but miss the exact spec
  Hybrid:  BM25 finds the exact chunk with "512", dense finds semantic context
           RRF puts the exact chunk at top, cross-encoder confirms it
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.db.chroma_client import get_or_create_collection
from app.rag.embedder import embed_query
from app.rag.bm25_index import get_index as get_bm25_index

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
    """A retrieved chunk with metadata and relevance score."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    chunk_index: int
    score: float        # cross-encoder score after final reranking
    metadata: dict
    retrieval_method: str = "hybrid"   # "dense" | "sparse" | "hybrid"


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a result at position rank.

    k=60 is the standard constant from the original RRF paper.
    Higher k = less aggressive discounting of lower-ranked results.
    """
    return 1.0 / (k + rank + 1)


def _reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],   # (chunk_id, dense_score)
    sparse_results: list[tuple[str, float]],  # (chunk_id, bm25_score)
    k: int = 60,
) -> list[tuple[str, float, str]]:
    """Merge dense and sparse ranked lists using RRF.

    Returns list of (chunk_id, rrf_score, retrieval_method) sorted by score.
    retrieval_method is "dense", "sparse", or "hybrid" (found in both).
    """
    rrf_scores: dict[str, float] = {}
    found_in: dict[str, set] = {}

    for rank, (cid, _) in enumerate(dense_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank, k)
        found_in.setdefault(cid, set()).add("dense")

    for rank, (cid, _) in enumerate(sparse_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank, k)
        found_in.setdefault(cid, set()).add("sparse")

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    for cid, score in merged:
        methods = found_in[cid]
        method = "hybrid" if len(methods) > 1 else list(methods)[0]
        result.append((cid, score, method))

    return result


def _fetch_chunks_by_ids(
    chunk_ids: list[str],
    client_id: str,
    existing: dict[str, "RetrievedChunk"],
) -> dict[str, "RetrievedChunk"]:
    """Fetch chunks from ChromaDB by ID.

    Only fetches IDs not already in existing — avoids redundant DB calls.
    """
    missing = [cid for cid in chunk_ids if cid not in existing]
    if not missing:
        return existing

    try:
        collection = get_or_create_collection(client_id)
        result = collection.get(
            ids=missing,
            include=["documents", "metadatas"],
        )
        fetched = dict(existing)
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            fetched[cid] = RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                source=meta.get("source", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=0.0,
                metadata=meta,
                retrieval_method="sparse",
            )
        return fetched
    except Exception as exc:
        logger.warning("Failed to fetch chunks by ID: %s", exc)
        return existing


def _rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Cross-encoder reranking on the merged candidate pool.

    Scores (query, raw_chunk_text) pairs for true relevance.
    Returns top_k sorted by score descending.
    """
    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()
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
        "Cross-encoder reranked %d → %d. Top score: %.4f",
        len(candidates), len(result),
        result[0].score if result else 0.0,
    )
    return result


def retrieve(
    query: str,
    client_id: str,
    coarse_k: int | None = None,
    fine_k: int | None = None,
) -> list[RetrievedChunk]:
    """Hybrid retrieval: Dense + BM25 → RRF → Cross-encoder reranking.

    Stage 1A — Dense retrieval from ChromaDB (top coarse_k)
    Stage 1B — BM25 keyword retrieval (top coarse_k)
    Stage 2  — RRF merges both lists into single ranked pool
    Stage 3  — Cross-encoder reranks merged pool, returns top fine_k
    """
    cfg = get_settings()
    coarse_k = coarse_k or cfg.coarse_k
    fine_k = fine_k or cfg.fine_k

    collection = get_or_create_collection(client_id)
    n_docs = collection.count()
    if n_docs == 0:
        logger.warning("Collection for client '%s' is empty.", client_id)
        return []

    actual_k = min(coarse_k, n_docs)

    # ── Stage 1A: Dense retrieval ─────────────────────────────────────────────
    query_vec = embed_query(query)
    dense_result = collection.query(
        query_embeddings=[query_vec],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    dense_ids = dense_result["ids"][0]
    dense_docs = dense_result["documents"][0]
    dense_metas = dense_result["metadatas"][0]
    dense_dists = dense_result["distances"][0]

    # Build chunk map from dense results (avoid re-fetching these from DB)
    chunk_map: dict[str, RetrievedChunk] = {}
    dense_ranked: list[tuple[str, float]] = []

    for cid, doc, meta, dist in zip(dense_ids, dense_docs, dense_metas, dense_dists):
        similarity = 1.0 - dist
        chunk_map[cid] = RetrievedChunk(
            chunk_id=cid,
            doc_id=meta.get("doc_id", ""),
            text=doc,
            source=meta.get("source", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            score=similarity,
            metadata=meta,
            retrieval_method="dense",
        )
        dense_ranked.append((cid, similarity))

    logger.debug(
        "Dense retrieval: %d candidates (client=%s)", len(dense_ranked), client_id
    )

    # ── Stage 1B: BM25 sparse retrieval ──────────────────────────────────────
    bm25_index = get_bm25_index(client_id)
    sparse_ranked = bm25_index.search(query, top_k=actual_k)

    # Fetch any BM25 chunks not already in dense results
    bm25_ids = [cid for cid, _ in sparse_ranked]
    chunk_map = _fetch_chunks_by_ids(bm25_ids, client_id, chunk_map)

    logger.debug(
        "BM25 retrieval: %d candidates with nonzero score (client=%s)",
        len(sparse_ranked), client_id,
    )

    # ── Stage 2: RRF fusion ───────────────────────────────────────────────────
    merged = _reciprocal_rank_fusion(dense_ranked, sparse_ranked)

    # Build ordered candidate list from merged ranking
    candidates: list[RetrievedChunk] = []
    for cid, rrf_score, method in merged:
        if cid in chunk_map:
            chunk = chunk_map[cid]
            chunk.retrieval_method = method
            candidates.append(chunk)

    # Log hybrid vs single-source stats
    hybrid_count = sum(1 for _, _, m in merged if m == "hybrid")
    dense_only = sum(1 for _, _, m in merged if m == "dense")
    sparse_only = sum(1 for _, _, m in merged if m == "sparse")
    logger.info(
        "RRF merged %d candidates: hybrid=%d dense_only=%d sparse_only=%d",
        len(candidates), hybrid_count, dense_only, sparse_only,
    )

    # ── Stage 3: Cross-encoder reranking ─────────────────────────────────────
    reranked = _rerank(query, candidates, top_k=fine_k)
    logger.debug("Final: %d chunks after cross-encoder reranking.", len(reranked))

    return reranked