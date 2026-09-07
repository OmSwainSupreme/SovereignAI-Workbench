"""Local vector stores for the RAG subsystem (Phase 5B).

A :class:`VectorStore` is the component that holds the embedded chunks and
answers similarity queries. The interface is intentionally thin so that a
future production deployment can swap in Qdrant, FAISS, or another specialised
store without changing the :class:`core.rag.retriever.Retriever` or
:class:`core.rag.knowledge_base.KnowledgeBase`.

For this phase we ship :class:`SimpleVectorStore` — a pure-Python,
in-memory store with cosine-similarity search. No numpy, no external
dependencies, no server process. It is fast enough for development and
tests on the Ryzen 3 laptop.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Final

from core.rag.types import ChunkMetadata


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vectors with norm below this threshold are treated as zero-vectors.
# A zero-vector cannot produce a meaningful cosine similarity.
_ZERO_NORM_EPSILON: Final[float] = 1e-10


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract vector store.

    The store holds a collection of vector entries, each keyed by a
    ``chunk_id`` (a stable string identifier). It supports basic CRUD
    operations and ranked similarity search.

    The similarity metric is cosine similarity. All vectors are stored
    unit-normalised so that similarity = dot product of the vectors.
    """

    @abstractmethod
    def add(self, chunk_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Add a new vector entry.

        Args:
            chunk_id: Unique identifier for this chunk. Must not already
                exist in the store.
            vector: The embedding vector. Must have the expected dimension.
            metadata: Arbitrary metadata to associate with this entry
                (e.g. :class:`ChunkMetadata`).

        Raises:
            ValueError: if ``chunk_id`` already exists.
        """

    @abstractmethod
    def upsert(self, chunk_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Insert or replace a vector entry.

        Unlike :meth:`add`, this is always safe to call regardless of
        whether the ``chunk_id`` already exists.
        """

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """Delete a vector entry by chunk_id.

        Returns:
            True if the entry was deleted; False if it was not present.
        """

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, list[float], dict[str, Any], float]]:
        """Find the top-K most similar entries.

        Args:
            query_vector: The query embedding. Must have the expected dimension.
            top_k: Maximum number of results to return.
            min_score: Minimum cosine similarity threshold. Entries with a
                score below this threshold are excluded from results.

        Returns:
            A list of ``(chunk_id, vector, metadata, score)`` tuples,
            sorted by descending score. May be shorter than ``top_k`` if
            fewer entries pass ``min_score``.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the store."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of entries in the store."""

    @abstractmethod
    def contains(self, chunk_id: str) -> bool:
        """Return True if a chunk_id is present in the store."""


# ---------------------------------------------------------------------------
# In-process implementation
# ---------------------------------------------------------------------------


class SimpleVectorStore(VectorStore):
    """A simple in-memory vector store with cosine-similarity search.

    All vectors are stored unit-normalised; search uses a dot product,
    which equals cosine similarity for unit vectors.

    Performance notes for the Ryzen 3 (6 GB RAM, no GPU):

    * ``O(n)`` scan over all entries per query. Acceptable for thousands
      of vectors, not hundreds of thousands.
    * Pure Python — no numpy means slightly higher per-op overhead but
      zero installation friction.
    * ``top_k`` sorting is done with Python's built-in ``sort`` (Timsort).

    The store is NOT thread-safe. Wrap with an async lock at the caller
    level if concurrent access is needed.
    """

    __slots__ = ("_dimension", "_entries", "_index")

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self._dimension = dimension
        # _entries: chunk_id -> (normalized_vector, metadata)
        self._entries: dict[str, tuple[list[float], dict[str, Any]]] = {}
        # _index: chunk_id -> normalized_vector (for fast search iteration)
        self._index: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _normalize(self, vector: list[float]) -> list[float]:
        """Return the L2-normalised copy of vector."""
        norm = math.sqrt(sum(v * v for v in vector))
        if norm < _ZERO_NORM_EPSILON:
            return vector
        return [v / norm for v in vector]

    def _dot(self, a: list[float], b: list[float]) -> float:
        """Dot product of two equal-length vectors."""
        return sum(av * bv for av, bv in zip(a, b))

    def add(self, chunk_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        if chunk_id in self._entries:
            raise ValueError(f"chunk_id already exists: {chunk_id!r}")
        self.upsert(chunk_id, vector, metadata)

    def upsert(
        self, chunk_id: str, vector: list[float], metadata: dict[str, Any]
    ) -> None:
        normalized = self._normalize(vector)
        self._entries[chunk_id] = (normalized, dict(metadata))
        self._index[chunk_id] = normalized

    def delete(self, chunk_id: str) -> bool:
        if chunk_id not in self._entries:
            return False
        del self._entries[chunk_id]
        del self._index[chunk_id]
        return True

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, list[float], dict[str, Any], float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._entries:
            return []

        q_norm = self._normalize(query_vector)
        scored: list[tuple[float, str]] = [
            (self._dot(q_norm, vec), chunk_id)
            for chunk_id, vec in self._index.items()
        ]
        # Sort descending by score
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[tuple[str, list[float], dict[str, Any], float]] = []
        for score, chunk_id in scored:
            if score < min_score:
                break
            if len(results) >= top_k:
                break
            vec, meta = self._entries[chunk_id]
            results.append((chunk_id, vec, meta, score))

        return results

    def clear(self) -> None:
        self._entries.clear()
        self._index.clear()

    def count(self) -> int:
        return len(self._entries)

    def contains(self, chunk_id: str) -> bool:
        return chunk_id in self._entries
