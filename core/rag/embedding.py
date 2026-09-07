"""Embedding providers for the RAG subsystem (Phase 5B).

An :class:`EmbeddingProvider` converts text into dense vector representations.
The interface is deliberately abstract — the actual model (ONNX sentence
transformers, llama.cpp, Ollama, OpenAI-compatible endpoint, etc.) is
swapped via a concrete implementation.

For this phase we ship a :class:`FakeEmbeddingProvider` that produces
deterministic pseudo-embeddings suitable for testing and development on
resource-constrained hardware.
"""
from __future__ import annotations

import hashlib
import math
import random
from abc import ABC, abstractmethod
from typing import Final


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract embedding provider.

    Implementations expose two methods: ``embed`` for a single text, and
    ``embed_batch`` for many texts. The batch method exists so that callers
    can pipeline the ingestion of many chunks in a single call.

    The interface does not specify a particular vector dimension or
    similarity metric — those are details of the concrete implementation.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text as a vector.

        Args:
            text: The text to embed. May be any non-empty string.

        Returns:
            A list of floats representing the text in embedding space.
            The length of the list is the ``dimension`` of this provider.

        Raises:
            ValueError: if ``text`` is empty.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts as vectors.

        The default implementation calls ``embed`` for each text in order.
        Subclasses may override this with a more efficient batch call.

        Args:
            texts: The texts to embed. Empty strings within the list
                are handled by the single-call implementation.

        Returns:
            A list of vectors, one per input text, in the same order.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of vectors produced by this provider."""


# ---------------------------------------------------------------------------
# Fake / deterministic provider for tests
# ---------------------------------------------------------------------------


# Smallest prime ≥ 128 — used to initialise the pseudo-random generator
# with a well-defined state that is independent of Python's internal
# float encoding.
_EMBEDDING_DIM: Final[int] = 128


class FakeEmbeddingProvider(EmbeddingProvider):
    """A deterministic fake embedding provider for testing and development.

    Each unique text string produces the same vector every time (the
    generator is seeded from a hash of the text). The resulting vectors
    are unit-normalised random directions in :math:`\\mathbb{R}^{128}`.

    This is NOT suitable for production — the vectors have no semantic
    meaning. It exists solely to exercise the RAG pipeline on the Ryzen 3
    laptop without downloading a real model.

    Key properties:

    * **Deterministic**: same input → same output across processes and runs.
    * **Seeded per text**: the RNG is seeded from ``SHA256(text + seed)``,
      so different texts produce different vectors.
    * **Unit-normalised**: all output vectors have L2 norm of 1.0, which
      means cosine similarity simplifies to a dot product.
    """

    __slots__ = ("_dimension", "_seed")

    def __init__(
        self,
        dimension: int = _EMBEDDING_DIM,
        seed: int = 42,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self._dimension = dimension
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        if not text:
            raise ValueError("text must not be empty")
        # Derive a per-text seed from the text itself and the class-level seed.
        seed_bytes = hashlib.sha256(
            f"{self._seed}:{text}".encode("utf-8")
        ).digest()
        # Use the first 8 bytes of the hash as a numeric seed.
        text_seed = int.from_bytes(seed_bytes[:8], "big")
        rng = random.Random(text_seed)

        # Generate random components and L2-normalise.
        components = [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]
        norm = math.sqrt(sum(c * c for c in components))
        if norm == 0.0:
            # Extremely unlikely: all random values happened to be zero.
            # Fall back to the unit vector along the first axis.
            components = [1.0] + [0.0] * (self._dimension - 1)
            norm = 1.0

        return [c / norm for c in components]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
