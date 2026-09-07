"""OCR → RAG bridge (Phase 5C).

This module provides a single helper, :func:`ocr_to_document`, that
converts OCR output into a :class:`core.rag.types.Document` so the
existing RAG pipeline (chunker → embedding provider → vector store)
can ingest OCR-extracted text without any changes.

The OCR subsystem remains responsible for OCR; the RAG subsystem remains
responsible for chunking/embedding. The bridge is a thin adapter.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from core.rag.types import Document
from core.vision.types import OCRResult


def ocr_to_document(
    results: Sequence[OCRResult],
    *,
    document_id: str,
    filename: str,
    source_path: str,
) -> Document:
    """Convert one or more OCR results into a RAG :class:`Document`.

    The ``full_text`` from each :class:`OCRResult` is joined with a
    double newline (matching the RAG chunker's default block separator).
    The document's metadata records how many pages it was built from,
    and (where available) the per-page confidence and language.

    Args:
        results: The OCR results to merge, in order.
        document_id: The RAG document ID. Use a stable identifier so
            re-ingestion is idempotent.
        filename: The document's filename (no path).
        source_path: A workspace-relative source identifier.

    Returns:
        A :class:`Document` ready to be passed to a
        :class:`core.rag.chunker.TextChunker`.
    """
    if not results:
        text = ""
    else:
        # Join pages with a clear page break marker. The RAG chunker
        # treats ``\n\n`` as a paragraph break, so two newlines suffice.
        text = "\n\n".join(r.full_text for r in results)

    # Build a compact, safe metadata dict for the RAG document. We
    # deliberately do NOT include the full OCR text here — it lives in
    # the document's ``text`` field.
    page_count = len(results)
    avg_confidence: float | None
    if results and all(r.confidence is not None for r in results):
        avg_confidence = sum(r.confidence or 0.0 for r in results) / page_count
    else:
        avg_confidence = None

    languages = sorted({r.language for r in results if r.language})
    metadata = {
        "source_kind": "ocr",
        "page_count": page_count,
        "average_confidence": avg_confidence,
        "languages": languages,
    }

    return Document(
        document_id=document_id,
        filename=filename,
        source_path=source_path,
        content_type="text/plain",  # OCR output is normalised text
        text=text,
        metadata=metadata,
    )


def ocr_results_to_full_text(results: Iterable[OCRResult]) -> str:
    """Concatenate OCR result texts into a single string.

    This is a convenience for callers that just want the text and
    don't need a full :class:`Document`.
    """
    return "\n\n".join(r.full_text for r in results)
