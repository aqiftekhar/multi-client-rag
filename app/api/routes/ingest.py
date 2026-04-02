"""Document ingestion routes — supports raw text and PDF file uploads."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.api.models import IngestRequest, IngestResponse
from app.clients import manager
from app.db.chroma_client import delete_collection
from app.ingestion.intake import ingest_document
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_MAX_PDF_MB = 50


def _extract_pdf_text(pdf_bytes: bytes, filename: str) -> str:
    """Extract text, tables, and images from a PDF.

    Per page strategy:
      1. Extract direct text (fast, accurate for digital PDFs)
      2. Extract tables → convert to readable text format
      3. Extract embedded images → OCR each one
      4. If page has no text at all (scanned) → OCR the whole rendered page
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF not installed. Run: pip install pymupdf",
        )

    try:
        import pytesseract
        from PIL import Image
        import io
        ocr_available = True
    except ImportError:
        ocr_available = False
        logger.warning("pytesseract/Pillow not installed — image OCR disabled.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not open PDF: {exc}")

    logger.info("Extracting PDF '%s' — %d pages", filename, len(doc))

    all_pages: list[str] = []
    MIN_TEXT_CHARS = 30
    MIN_IMAGE_SIZE = 100  # skip tiny images (icons, bullets)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_label = f"[Page {page_num + 1}]"
        page_parts: list[str] = []

        # ── 1. Direct text extraction ─────────────────────────────────────────
        direct_text = page.get_text("text").strip()
        if direct_text:
            page_parts.append(direct_text)

        # ── 2. Table extraction ───────────────────────────────────────────────
        # PyMuPDF 1.23+ has find_tables()
        try:
            tables = page.find_tables()
            for table_index, table in enumerate(tables, 1):
                rows = table.extract()
                if not rows:
                    continue

                # Convert table rows to readable plain text
                table_lines = [f"[Table {table_index}]"]
                for row in rows:
                    # Clean None values and strip whitespace
                    cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
                    table_lines.append(" | ".join(cleaned))

                table_text = "\n".join(table_lines)
                # Only add if table has actual content
                if any(cell for row in rows for cell in row if cell):
                    page_parts.append(table_text)
                    logger.debug(
                        "Page %d: extracted table %d (%d rows)",
                        page_num + 1, table_index, len(rows)
                    )
        except AttributeError:
            # find_tables() not available in older PyMuPDF versions
            logger.debug("Table extraction not available — upgrade pymupdf if needed.")
        except Exception as exc:
            logger.warning("Table extraction failed on page %d: %s", page_num + 1, exc)

        # ── 3. Image extraction + OCR ─────────────────────────────────────────
        if ocr_available:
            image_list = page.get_images(full=True)
            for img_ref in image_list:
                xref = img_ref[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    w, h = pil_image.size

                    # Skip tiny images — they're likely icons or decorations
                    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                        continue

                    ocr_text = pytesseract.image_to_string(pil_image, lang="eng").strip()
                    if ocr_text and len(ocr_text) > 20:
                        page_parts.append(f"[Image Text]\n{ocr_text}")
                        logger.debug(
                            "Page %d: OCR'd image %dx%d → %d chars",
                            page_num + 1, w, h, len(ocr_text)
                        )
                except Exception as exc:
                    logger.warning("Image OCR failed on page %d: %s", page_num + 1, exc)

        # ── 4. Scanned page fallback — OCR the entire rendered page ──────────
        # Triggered when: no direct text AND no embedded images extracted text
        if ocr_available and len(direct_text) < MIN_TEXT_CHARS and not page.get_images():
            try:
                # Render page at 2x zoom for better OCR accuracy (~144 DPI)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pil_page = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                ocr_text = pytesseract.image_to_string(pil_page, lang="eng").strip()
                if ocr_text:
                    page_parts.append(f"[Scanned Page OCR]\n{ocr_text}")
                    logger.info("Page %d: full-page OCR returned %d chars", page_num + 1, len(ocr_text))
                else:
                    logger.warning("Page %d: OCR returned nothing — image may be too low quality", page_num + 1)
            except Exception as exc:
                logger.warning("Full-page OCR failed on page %d: %s", page_num + 1, exc)

        # ── Combine page parts ────────────────────────────────────────────────
        if page_parts:
            all_pages.append(f"{page_label}\n" + "\n\n".join(page_parts))

    doc.close()

    full_text = "\n\n".join(all_pages)

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No content could be extracted from this PDF. "
                "It may be corrupt, password-protected, or contain "
                "only non-readable content."
            ),
        )

    logger.info(
        "PDF extraction complete: %d pages, %d total chars",
        len(all_pages), len(full_text)
    )
    return full_text


@router.post("/", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Ingest a raw text document for a registered client."""
    if not manager.exists(req.client_id):
        raise HTTPException(
            status_code=404,
            detail=f"Client '{req.client_id}' not found. Register it first via POST /clients/.",
        )

    result = ingest_document(
        text=req.text,
        client_id=req.client_id,
        source=req.source,
        doc_id=req.doc_id,
        extra_metadata=req.extra_metadata,
    )

    return IngestResponse(
        doc_id=result.doc_id,
        client_id=result.client_id,
        total_chunks=result.total_chunks,
        stored_chunks=result.stored_chunks,
        duplicate_chunks=result.duplicate_chunks,
        anomalous_chunks=result.anomalous_chunks,
        errors=result.errors,
    )

@router.get("/{client_id}/documents")
def list_documents(client_id: str) -> dict:
    """List all ingested documents for a client with chunk counts and metadata."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")

    try:
        from app.db.chroma_client import get_or_create_collection
        collection = get_or_create_collection(client_id)
        if collection.count() == 0:
            return {"client_id": client_id, "documents": [], "total_chunks": 0}

        result = collection.get(include=["metadatas"])
        metadatas = result["metadatas"]

        # Group chunks by doc_id
        docs: dict[str, dict] = {}
        for meta in metadatas:
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", "unknown"),
                    "ingested_at": meta.get("ingested_at", ""),
                    "chunk_count": 0,
                    "file_type": meta.get("file_type", "text"),
                }
            docs[doc_id]["chunk_count"] += 1

        return {
            "client_id": client_id,
            "documents": sorted(
                docs.values(),
                key=lambda x: x["ingested_at"],
                reverse=True,
            ),
            "total_chunks": len(metadatas),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{client_id}/documents/{doc_id}")
def delete_document(client_id: str, doc_id: str) -> dict:
    """Delete all chunks belonging to a specific document."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")

    try:
        from app.db.chroma_client import get_or_create_collection
        collection = get_or_create_collection(client_id)
        result = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        if not result["ids"]:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

        chunk_count = len(result["ids"])
        collection.delete(ids=result["ids"])

        logger.info(
            "Deleted doc '%s' for client '%s': %d chunks removed.",
            doc_id, client_id, chunk_count,
        )
        return {
            "deleted": doc_id,
            "client_id": client_id,
            "chunks_removed": chunk_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(
    client_id: str = Form(...),
    file: UploadFile = File(...),
) -> IngestResponse:
    """Ingest a PDF file for a registered client.

    Extracts text from all pages and runs through the full ingestion pipeline.
    """
    if not manager.exists(client_id):
        raise HTTPException(
            status_code=404,
            detail=f"Client '{client_id}' not found. Register it first via POST /clients/.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted here.")

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(pdf_bytes) > _MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"PDF too large. Maximum size is {_MAX_PDF_MB}MB.",
        )

    full_text = _extract_pdf_text(pdf_bytes, file.filename)

    result = ingest_document(
        text=full_text,
        client_id=client_id,
        source=file.filename,
        extra_metadata={"file_type": "pdf", "original_filename": file.filename},
    )

    return IngestResponse(
        doc_id=result.doc_id,
        client_id=result.client_id,
        total_chunks=result.total_chunks,
        stored_chunks=result.stored_chunks,
        duplicate_chunks=result.duplicate_chunks,
        anomalous_chunks=result.anomalous_chunks,
        errors=result.errors,
    )


@router.delete("/{client_id}")
def clear_corpus(client_id: str) -> dict:
    """Delete all vectors for a client."""
    if not manager.exists(client_id):
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")
    delete_collection(client_id)
    return {"cleared": client_id}