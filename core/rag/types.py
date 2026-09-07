"""Framework-agnostic data types for the RAG subsystem (Phase 5B).

These types are deliberately plain dataclasses. They must not depend on
FastAPI, Pydantic, numpy, or any other framework. The application layer
(``backend/app``) may wrap them in Pydantic DTOs for HTTP transport, but
this module remains pure.

This mirrors the design convention of :mod:`core.agent.types` and
:mod:`core.tools.types`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current UTC time, timezone-aware."""
    return datetime.now(tz=timezone.utc)


def _stable_id(*parts: Any, length: int = 16) -> str:
    """Return a stable, content-derived identifier.

    The identifier is the first ``length`` hex characters of a SHA-256
    digest of the stringified parts. The function is deterministic for
    the same inputs.

    Args:
        *parts: Components of the identifier. Each part is stringified.
        length: How many hex characters to keep (max 64).

    Returns:
        A lowercase hex string of length ``length``.
    """
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """An ingested source document.

    A document is the result of parsing a single source file. The
    ``document_id`` is derived from the ``source_path`` and is stable
    across re-ingestions of the same file.

    The ``source_path`` is a *workspace-relative* path or any other
    application-meaningful identifier. Absolute host filesystem paths
    are NOT stored here.
    """

    document_id: str
    filename: str
    source_path: str
    content_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=_now_utc)


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkMetadata:
    """Source-grounding metadata for a single chunk.

    Carries the minimum information needed to cite the chunk later:
    which document it came from, the chunk's position within that
    document, and the character offsets in the original text. We do
    NOT invent page numbers — only metadata actually available from
    the source.
    """

    document_id: str
    filename: str
    content_type: str
    chunk_index: int
    char_start: int
    char_end: int
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Document chunk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of a document with its text and source metadata.

    The ``chunk_id`` is derived from the parent ``document_id`` and the
    ``chunk_index`` — re-chunking the same document produces the same
    chunk IDs.
    """

    chunk_id: str
    text: str
    metadata: ChunkMetadata

    @classmethod
    def from_document(
        cls,
        document: Document,
        text: str,
        chunk_index: int,
        char_start: int,
        char_end: int,
    ) -> "DocumentChunk":
        """Build a chunk from a document, with a deterministic ID."""
        chunk_id = _stable_id(document.document_id, chunk_index)
        meta = ChunkMetadata(
            document_id=document.document_id,
            filename=document.filename,
            content_type=document.content_type,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
        )
        return cls(chunk_id=chunk_id, text=text, metadata=meta)


# ---------------------------------------------------------------------------
# Retrieval result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retrieval query, with its similarity score."""

    chunk: DocumentChunk
    score: float
    rank: int  # 1-based


@dataclass(frozen=True)
class SearchResult:
    """The result of a single knowledge-base search.

    ``results`` is sorted by descending similarity. ``total_available``
    is the number of vectors the store searched over; it is independent
    of ``top_k`` so callers can report coverage.
    """

    query: str
    results: tuple[RetrievedChunk, ...] = field(default_factory=tuple)
    total_available: int = 0
    searched_at: datetime = field(default_factory=_now_utc)


@dataclass(frozen=True)
class IngestResult:
    """The result of a single document ingestion.

    ``chunk_ids`` is the list of vectors that were written to the
    store. ``was_new`` indicates whether this was a fresh ingestion or
    a re-ingestion that replaced prior vectors.
    """

    document_id: str
    filename: str
    num_chunks: int
    chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    was_new: bool = True
    ingested_at: datetime = field(default_factory=_now_utc)
