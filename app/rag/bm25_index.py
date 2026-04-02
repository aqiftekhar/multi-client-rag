"""BM25 per-client sparse index for hybrid search.

BM25 (Best Match 25) is a keyword-based ranking function that excels where
dense embeddings fail: exact terms, acronyms, proper nouns, IDs, and
technical keywords that get compressed or averaged away by embedding models.

Example where BM25 wins over dense:
  Query: "dmodel = 512"
  Dense: may retrieve semantically similar chunks about model sizes
  BM25:  finds the exact chunk containing "dmodel = 512" immediately

This index is per-client, in-memory, rebuilt from ChromaDB on startup.
Updated incrementally on every ingestion — no full rebuild needed.
"""

import logging
import re
from threading import Lock

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Per-client index store
_indices: dict[str, "ClientBM25Index"] = {}
_lock = Lock()


class ClientBM25Index:
    """BM25 index for a single client."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.chunk_ids: list[str] = []
        self.corpus: list[list[str]] = []   # tokenized texts parallel to chunk_ids
        self.bm25: BM25Okapi | None = None
        self._id_set: set[str] = set()      # fast duplicate check

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase + split on non-alphanumeric characters.

        Keeps numbers and hyphenated terms intact.
        Example: "feed-forward dmodel=512" → ["feed", "forward", "dmodel", "512"]
        """
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def add_chunks(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Add new chunks to the index incrementally.

        Skips chunks already present — safe to call on re-ingestion.
        """
        added = 0
        for cid, text in zip(chunk_ids, texts):
            if cid in self._id_set:
                continue
            self.chunk_ids.append(cid)
            self.corpus.append(self._tokenize(text))
            self._id_set.add(cid)
            added += 1

        if added > 0:
            self._rebuild()
            logger.debug(
                "BM25 index updated for client '%s': +%d chunks, total=%d",
                self.client_id, added, len(self.chunk_ids),
            )

    def _rebuild(self) -> None:
        """Rebuild the BM25 model from current corpus."""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) sorted by score descending."""
        if not self.bm25 or not self.chunk_ids:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        results = sorted(
            zip(self.chunk_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        # Filter zero scores — chunk has no token overlap with query
        results = [(cid, score) for cid, score in results if score > 0]
        return results[:top_k]

    @property
    def size(self) -> int:
        return len(self.chunk_ids)


# ── Public API ────────────────────────────────────────────────────────────────

def get_index(client_id: str) -> ClientBM25Index:
    """Return (or create) the BM25 index for client_id."""
    with _lock:
        if client_id not in _indices:
            _indices[client_id] = ClientBM25Index(client_id)
        return _indices[client_id]


def update_index(client_id: str, chunk_ids: list[str], raw_texts: list[str]) -> None:
    """Add chunks to the client's BM25 index.

    Called from intake.py after every ingestion.
    Uses raw_text only — no headers — same as embedding.
    """
    index = get_index(client_id)
    index.add_chunks(chunk_ids, raw_texts)


def rebuild_from_chromadb(client_id: str) -> int:
    """Rebuild BM25 index from ChromaDB for a client.

    Called on startup for each registered client so the index
    survives app restarts without re-ingesting documents.

    Returns number of chunks loaded.
    """
    try:
        from app.db.chroma_client import get_or_create_collection
        collection = get_or_create_collection(client_id)
        count = collection.count()
        if count == 0:
            logger.debug("BM25 rebuild: client '%s' has no chunks.", client_id)
            return 0

        # Fetch all chunks — use raw_text from metadata if available
        # otherwise fall back to stored document text
        result = collection.get(include=["documents", "metadatas"])
        ids = result["ids"]
        metadatas = result["metadatas"]
        documents = result["documents"]

        raw_texts = []
        for meta, doc in zip(metadatas, documents):
            raw_texts.append(meta.get("raw_text") or doc or "")

        index = get_index(client_id)
        index.add_chunks(ids, raw_texts)

        logger.info(
            "BM25 index rebuilt for client '%s': %d chunks loaded.",
            client_id, index.size,
        )
        return index.size

    except Exception as exc:
        logger.error("BM25 rebuild failed for client '%s': %s", client_id, exc)
        return 0