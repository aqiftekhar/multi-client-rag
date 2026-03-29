"""Sentence-aware text chunker with configurable token overlap."""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass

import tiktoken

from app.config import get_settings

logger = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    token_count: int


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight regex heuristic."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, doc_id: str | None = None) -> list[Chunk]:
    """Split *text* into overlapping token-bounded chunks.

    Strategy:
    - Accumulate sentences until the token budget is reached
    - Backtrack by `overlap` tokens before starting the next chunk
    - Each chunk is assigned a stable ID derived from doc_id + index
    """
    cfg = get_settings()
    doc_id = doc_id or str(uuid.uuid4())
    sentences = _split_sentences(text)

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        s_tokens = _token_count(sentence)

        # If a single sentence exceeds budget, force-split it
        if s_tokens > cfg.chunk_size:
            words = sentence.split()
            part = []
            part_tokens = 0
            for word in words:
                wt = _token_count(word)
                if part_tokens + wt > cfg.chunk_size and part:
                    chunk_text_str = " ".join(part)
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(doc_id, chunk_index),
                            doc_id=doc_id,
                            text=chunk_text_str,
                            chunk_index=chunk_index,
                            token_count=part_tokens,
                        )
                    )
                    chunk_index += 1
                    # Overlap: keep last N tokens
                    part = part[max(0, len(part) - cfg.chunk_overlap // 4):]
                    part_tokens = _token_count(" ".join(part))
                part.append(word)
                part_tokens += wt
            if part:
                current_sentences.extend(part)
                current_tokens += part_tokens
            continue

        if current_tokens + s_tokens > cfg.chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc_id, chunk_index),
                    doc_id=doc_id,
                    text=chunk_text_str,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                )
            )
            chunk_index += 1

            # Overlap: keep sentences worth ≤ chunk_overlap tokens
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                st = _token_count(s)
                if overlap_tokens + st <= cfg.chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += st
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += s_tokens

    # Final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc_id, chunk_index),
                doc_id=doc_id,
                text=chunk_text_str,
                chunk_index=chunk_index,
                token_count=current_tokens,
            )
        )

    logger.debug("Chunked doc '%s' into %d chunks.", doc_id, len(chunks))
    return chunks


def _chunk_id(doc_id: str, index: int) -> str:
    raw = f"{doc_id}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
