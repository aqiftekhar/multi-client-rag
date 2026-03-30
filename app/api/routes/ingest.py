"""Document ingestion routes — supports raw text and PDF file uploads."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.api.models import IngestRequest, IngestResponse
from app.clients import manager
from app.db.chroma_client import delete_collection
from app.ingestion.intake import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_MAX_PDF_MB = 50


def _extract_pdf_text(pdf_bytes: bytes, filename: str) -> str:
    """Extract all text from a PDF using PyMuPDF.

    Handles digital PDFs with selectable text.
    Each page is separated clearly so the chunker can work with it.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF not installed. Run: pip install pymupdf",
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not open PDF: {exc}")

    pages_text = []
    for i, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            pages_text.append(f"[Page {i}]\n{text}")

    doc.close()

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from this PDF. "
                "It may be a scanned image-only PDF. "
                "Please convert it to a text-based PDF first."
            ),
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