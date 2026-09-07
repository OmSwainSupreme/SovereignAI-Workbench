"""Text chunkers for the RAG subsystem (Phase 5B).

A chunker splits a :class:`~core.rag.types.Document` into a sequence of
:class:`~core.rag.types.DocumentChunk` objects. Chunks preserve source
metadata and positional offsets so they can be cited back to the original.

For this phase we use a deterministic character-count chunker with
configurable overlap. More sophisticated semantic chunking can be added
later by implementing :class:`TextChunker`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.rag.types import Document, DocumentChunk


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TextChunker(ABC):
    """Abstract text chunker.

    Implementations split a document into chunks. Every chunk carries a
    :class:`~core.rag.types.ChunkMetadata` with source provenance.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split a document into chunks.

        Args:
            document: The document to chunk.

        Returns:
            A list of :class:`DocumentChunk` objects, in order.
            Empty input produces an empty list.
        """


# ---------------------------------------------------------------------------
# Deterministic character-count chunker
# ---------------------------------------------------------------------------


class DeterministicChunker(TextChunker):
    """A deterministic character-count chunker with optional overlap.

    The chunker walks through the document text in fixed-size steps. Each
    chunk contains up to ``chunk_size`` characters. Consecutive chunks
    overlap by ``overlap`` characters, which helps retrieval yield
    coherent snippets.

    The chunking is fully deterministic:

    * The same document always produces the same chunk list.
    * Chunk IDs are derived from ``(document_id, chunk_index)`` via a
      stable hash, so they survive re-chunking.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
    """

    __slots__ = ("_chunk_size", "_overlap")

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap < 0:
            raise ValueError("overlap must not be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.text
        if not text:
            return []

        step = self._chunk_size - self._overlap
        chunks: list[DocumentChunk] = []
        index = 0
        pos = 0
        total = len(text)

        while pos < total:
            end = min(pos + self._chunk_size, total)
            chunk_text = text[pos:end]
            chunk = DocumentChunk.from_document(
                document=document,
                text=chunk_text,
                chunk_index=index,
                char_start=pos,
                char_end=end,
            )
            chunks.append(chunk)
            index += 1
            if step <= 0:
                # Degenerate case: overlap == chunk_size → infinite loop
                break
            pos += step

        return chunks
