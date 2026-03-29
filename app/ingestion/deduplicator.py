"""Deduplication for ingested chunks using exact hash and MinHash LSH."""

import hashlib
import logging
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

_NUM_PERM = 128
_SIMILARITY_THRESHOLD = 0.85


@dataclass
class DeduplicationStore:
    """In-memory (per-process) dedup store. Persists in ChromaDB metadata on restart."""

    exact_hashes: set[str] = field(default_factory=set)
    lsh: MinHashLSH = field(
        default_factory=lambda: MinHashLSH(threshold=_SIMILARITY_THRESHOLD, num_perm=_NUM_PERM)
    )
    _key_counter: int = field(default=0, init=False)

    def _text_to_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=_NUM_PERM)
        for token in text.lower().split():
            m.update(token.encode("utf-8"))
        return m

    def is_duplicate(self, text: str) -> bool:
        """Return True if *text* is an exact or near-duplicate of already-seen content."""
        exact = hashlib.sha256(text.encode()).hexdigest()
        if exact in self.exact_hashes:
            logger.debug("Exact duplicate detected.")
            return True

        mh = self._text_to_minhash(text)
        result = self.lsh.query(mh)
        if result:
            logger.debug("Near-duplicate detected (LSH matches: %s).", result)
            return True

        return False

    def add(self, text: str, key: str | None = None) -> None:
        """Register *text* so future calls to :meth:`is_duplicate` detect it."""
        exact = hashlib.sha256(text.encode()).hexdigest()
        self.exact_hashes.add(exact)

        mh = self._text_to_minhash(text)
        k = key or f"chunk_{self._key_counter}"
        self._key_counter += 1
        try:
            self.lsh.insert(k, mh)
        except ValueError:
            # Key already exists — not a problem
            pass


# Global per-client stores (keyed by client_id)
_stores: dict[str, DeduplicationStore] = {}


def get_store(client_id: str) -> DeduplicationStore:
    """Return (or create) the dedup store for *client_id*."""
    if client_id not in _stores:
        _stores[client_id] = DeduplicationStore()
    return _stores[client_id]


def deduplicate(texts: list[str], client_id: str) -> list[tuple[str, bool]]:
    """Filter *texts* for a client. Returns list of (text, is_duplicate) tuples."""
    store = get_store(client_id)
    results = []
    for text in texts:
        dup = store.is_duplicate(text)
        if not dup:
            store.add(text)
        results.append((text, dup))
    unique_count = sum(1 for _, d in results if not d)
    logger.info("Dedup: %d/%d unique chunks for client '%s'", unique_count, len(texts), client_id)
    return results
