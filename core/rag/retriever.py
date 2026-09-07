"""Retriever for the RAG subsystem (Phase 5B).

The retriever bridges the vector store and the embedding provider. It accepts
a natural-language query, converts it to an embedding, searches the vector store
for the top-K most similar chunks, and returns ranked retrieval results with
full source metadata.

The retriever is agnostic about which embedding model or vector store
implementation is used.
"""
from __future__ import annotations

import logging

from core.rag.embedding import EmbeddingProvider
from core.rag.types import (
    DocumentChunk,
    RetrievedChunk,
    SearchResult,
    _now_utc,
)
from core.rag.vector_store import VectorStore


_logger = logging.getLogger("sovereign-ai.rag.retriever")


class Retriever:
    """A RAG retriever built on an embedding provider and a vector store.

    The retriever does not own its dependencies — it receives them via
    the constructor, making it easy to swap either one.

    Example::

        retriever = Retriever(
            vector_store=SimpleVectorStore(dimension=128),
            embedding_provider=FakeEmbeddingProvider(dimension=128),
        )
        results = retriever.retrieve("safety procedures", top_k=5)
    """

    __slots__ = ("_vector_store", "_embedding_provider")

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> SearchResult:
        """Search the knowledge base for chunks relevant to the query.

        Args:
            query: The natural-language search query.
            top_k: Maximum number of chunks to return.
            min_score: Minimum cosine similarity threshold.

        Returns:
            A :class:`SearchResult` containing ranked :class:`RetrievedChunk`
            objects with source metadata and similarity scores.
        """
        if not query:
            return SearchResult(query=query, results=(), total_available=0)

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if min_score < 0.0 or min_score > 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")

        _logger.debug(
            "retriever.search  top_k=%d  min_score=%.2f  store_count=%d",
            top_k,
            min_score,
            self._vector_store.count(),
        )

        # 1. Embed the query
        query_vector = self._embedding_provider.embed(query)

        # 2. Search the vector store
        raw_results = self._vector_store.search(
            query_vector,
            top_k=top_k,
            min_score=min_score,
        )

        # 3. Wrap results as RetrievedChunk objects
        retrieved: list[RetrievedChunk] = []
        for rank, (chunk_id, _vector, metadata, score) in enumerate(
            raw_results, start=1
        ):
            chunk = self._chunk_from_metadata(chunk_id, metadata)
            retrieved.append(
                RetrievedChunk(chunk=chunk, score=score, rank=rank)
            )

        total_available = self._vector_store.count()
        return SearchResult(
            query=query,
            results=tuple(retrieved),
            total_available=total_available,
            searched_at=_now_utc(),
        )

    # ----------------------------------------------------------------- Internals

    def _chunk_from_metadata(
        self, chunk_id: str, metadata: dict
    ) -> DocumentChunk:
        """Reconstruct a minimal DocumentChunk from stored metadata.

        The vector store only holds metadata, not the full DocumentChunk
        object. This method rebuilds the chunk from its metadata dict.

        We import lazily to avoid circular imports.
        """
        from core.rag.types import ChunkMetadata

        meta = ChunkMetadata(
            document_id=metadata["document_id"],
            filename=metadata["filename"],
            content_type=metadata["content_type"],
            chunk_index=metadata["chunk_index"],
            char_start=metadata["char_start"],
            char_end=metadata["char_end"],
        )
        return DocumentChunk(
            chunk_id=chunk_id,
            text=metadata.get("text", ""),
            metadata=meta,
        )
