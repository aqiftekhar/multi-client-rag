"""Document intake pipeline: clean → dedup → anomaly check → chunk → embed → store."""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.ingestion.cleaner import clean
from app.ingestion.deduplicator import deduplicate
from app.ingestion.anomaly_detector import filter_anomalies
from app.rag.chunker import chunk_text, build_context_header, Chunk
from app.rag.embedder import embed_chunks
from app.db.chroma_client import get_or_create_collection

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""

    doc_id: str
    client_id: str
    total_chunks: int
    stored_chunks: int
    duplicate_chunks: int
    anomalous_chunks: int
    errors: list[str] = field(default_factory=list)


def ingest_document(
    text: str,
    client_id: str,
    source: str = "unknown",
    doc_id: str | None = None,
    extra_metadata: dict | None = None,
) -> IngestionResult:
    """Run the full intake pipeline for a single document.

    Pipeline:
      clean → chunk → dedup → anomaly filter → embed → store in ChromaDB
    """
    doc_id = doc_id or str(uuid.uuid4())
    extra_metadata = extra_metadata or {}
    errors: list[str] = []

    # 1. Clean
    cleaned = clean(text)
    if not cleaned:
        return IngestionResult(
            doc_id=doc_id,
            client_id=client_id,
            total_chunks=0,
            stored_chunks=0,
            duplicate_chunks=0,
            anomalous_chunks=0,
            errors=["document_empty_after_cleaning"],
        )

    # 2. Chunk
    chunks: list[Chunk] = chunk_text(cleaned, doc_id=doc_id)
    raw_texts = [c.text for c in chunks]

    # 3. Anomaly filter
    clean_texts, anomaly_reports = filter_anomalies(raw_texts)
    anomalous_count = sum(1 for r in anomaly_reports if r.is_anomalous)

    # Map back to chunks (only non-anomalous)
    anomalous_set = {r.chunk_text for r in anomaly_reports if r.is_anomalous}
    clean_chunks = [c for c in chunks if c.text not in anomalous_set]

    # 4. Dedup
    dedup_pairs = deduplicate([c.text for c in clean_chunks], client_id=client_id)
    unique_chunks = [
        chunk for (chunk, (_, is_dup)) in zip(clean_chunks, dedup_pairs) if not is_dup
    ]
    duplicate_count = len(clean_chunks) - len(unique_chunks)

    if not unique_chunks:
        logger.info(
            "No unique chunks after dedup for doc '%s' client '%s'.", doc_id, client_id
        )
        return IngestionResult(
            doc_id=doc_id,
            client_id=client_id,
            total_chunks=len(chunks),
            stored_chunks=0,
            duplicate_chunks=duplicate_count,
            anomalous_chunks=anomalous_count,
        )

    # 5. Build contextual headers — done after dedup so index stays meaningful
    for chunk in unique_chunks:
        chunk.context_header = build_context_header(
            source=source,
            doc_id=doc_id,
            chunk_index=chunk.chunk_index,
            total_chunks=len(chunks),
        )

    # 6. Embed RAW text only — header must NOT be embedded.
    #    Embedding the header would dilute the semantic signal.
    embeddings = embed_chunks([c.text for c in unique_chunks])

    # 7. Store in ChromaDB
    #    document  = header + text  → what the LLM receives
    #    embedding = raw text       → what retrieval uses
    collection = get_or_create_collection(client_id)
    now = datetime.now(timezone.utc).isoformat()

    ids, docs, metadatas, vecs = [], [], [], []
    for chunk, embedding in zip(unique_chunks, embeddings):
        chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:16]
        stored_text = f"{chunk.context_header}\n\n{chunk.text}"

        ids.append(chunk.chunk_id)
        docs.append(stored_text)
        metadatas.append(
            {
                "doc_id": doc_id,
                "client_id": client_id,
                "source": source,
                "chunk_index": chunk.chunk_index,
                "chunk_hash": chunk_hash,
                "ingested_at": now,
                "context_header": chunk.context_header,
                **extra_metadata,
            }
        )
        vecs.append(embedding)

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=vecs)

    logger.info(
        "Ingested doc '%s' for client '%s': %d stored, %d dupes, %d anomalous.",
        doc_id,
        client_id,
        len(unique_chunks),
        duplicate_count,
        anomalous_count,
    )

    return IngestionResult(
        doc_id=doc_id,
        client_id=client_id,
        total_chunks=len(chunks),
        stored_chunks=len(unique_chunks),
        duplicate_chunks=duplicate_count,
        anomalous_chunks=anomalous_count,
        errors=errors,
    )
