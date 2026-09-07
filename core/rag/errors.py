"""Exception hierarchy for the RAG subsystem.

These errors are part of the public surface of the RAG layer. The
application layer should catch :class:`RAGError` and translate to a
domain-specific error or HTTP response as appropriate.
"""
from __future__ import annotations

from typing import Optional


class RAGError(Exception):
    """Base class for all RAG errors."""


class RAGConfigurationError(RAGError):
    """The RAG subsystem is misconfigured.

    Examples: missing config file, invalid values, unknown embedding
    provider name.
    """


class RAGIngestError(RAGError):
    """A document could not be ingested.

    Attributes:
        path: The workspace-relative path of the failing document.
        reason: A short diagnostic (safe to surface to end users).
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to ingest {path!r}: {reason}")


class RAGSearchError(RAGError):
    """A search could not be completed.

    Attributes:
        reason: A short diagnostic explaining the failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Search failed: {reason}")
