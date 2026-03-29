"""Embedding module using sentence-transformers (local, no API key required)."""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache()
def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    model_name = get_settings().embed_model
    logger.info("Loading embedding model '%s'...", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings.

    Returns a list of float vectors, one per input text.
    """
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Alias for :func:`embed_texts` — kept for semantic clarity at call sites."""
    return embed_texts(chunks)


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
