"""HyDE — Hypothetical Document Embeddings.

Problem solved:
  User queries are short and informal: "what are the performance numbers?"
  Document chunks are long and formal: "achieves 28.4 BLEU on WMT 2014"
  Their embeddings don't overlap well — retrieval misses the right chunk.

Solution:
  Ask the LLM to generate a hypothetical answer to the query.
  A hypothetical answer uses the same vocabulary and style as real document text.
  Embed the hypothetical answer instead of the raw query.
  Use that embedding for retrieval — much better overlap with real chunks.

When HyDE is used:
  Only for queries where the raw query embedding retrieves low-scoring results.
  Falls back to raw query embedding if HyDE generation fails or is slow.

Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""

import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

_HYDE_PROMPT = """Generate a short passage (2-3 sentences) that would directly answer this question.
Write as if you are extracting from a technical document.
Be specific and use technical language.
Do NOT say you don't know — generate a plausible answer.
Do NOT add any preamble or explanation — just the passage.

Question: {query}

Passage:"""


def generate_hypothetical_document(
    query: str,
    ollama_host: str,
    model: str,
    timeout: float = 20.0,
) -> str | None:
    """Generate a hypothetical document passage for the given query.

    Returns the generated text, or None if generation fails or times out.
    Keeps temperature high (0.7) to generate diverse, document-like text.
    """
    start = time.time()
    try:
        response = httpx.post(
            f"{ollama_host}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": _HYDE_PROMPT.format(query=query),
                    }
                ],
                "options": {
                    "temperature": 0.7,   # higher temp for diverse hypotheticals
                    "num_predict": 150,   # short passage only
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()

        # Strip thinking blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        elapsed = time.time() - start
        logger.info(
            "HyDE generated in %.2fs for query='%.40s'",
            elapsed, query,
        )
        return raw if raw else None

    except Exception as exc:
        elapsed = time.time() - start
        logger.warning(
            "HyDE generation failed in %.2fs: %s — falling back to raw query",
            elapsed, exc,
        )
        return None


def get_hyde_embedding(
    query: str,
    ollama_host: str,
    model: str,
) -> list[float] | None:
    """Generate a HyDE embedding for the query.

    Returns the embedding of the hypothetical document,
    or None if generation fails (caller should fall back to raw query embedding).
    """
    from app.rag.embedder import embed_query
    hypothetical = generate_hypothetical_document(query, ollama_host, model)
    if not hypothetical:
        return None
    return embed_query(hypothetical)