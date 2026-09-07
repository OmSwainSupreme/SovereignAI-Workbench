"""Document parsers for the RAG subsystem (Phase 5B).

A parser converts a raw source (workspace-relative path + raw text) into a
normalised :class:`~core.rag.types.Document` object. Parsers are
swappable — a future PDF parser or OCR processor would implement the same
interface.

For this phase only plain text files are supported.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import Mapping

from core.rag.types import Document, _stable_id


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DocumentParser(ABC):
    """Abstract document parser.

    Implementations take raw text and a source identifier and return a
    normalised :class:`Document`. The source identifier is typically a
    workspace-relative path, but the parser itself is agnostic about its
    meaning.
    """

    @abstractmethod
    def parse(self, text: str, source: str) -> Document:
        """Parse raw text into a Document.

        Args:
            text: The raw text content of the document.
            source: An application-meaningful identifier for the source
                (e.g. a workspace-relative path or URL).

        Returns:
            A :class:`Document` with a stable ``document_id``.
        """

    @property
    @abstractmethod
    def supported_content_type(self) -> str:
        """The MIME-like content type this parser produces."""

    @property
    def extra_metadata(self) -> Mapping[str, object]:
        """Extra metadata to attach to every parsed document (override in subclass)."""
        return {}


# ---------------------------------------------------------------------------
# Plain-text parser
# ---------------------------------------------------------------------------


class PlainTextParser(DocumentParser):
    """Parser for plain-text documents.

    The ``document_id`` is derived from the source path so that re-parsing
    the same file always produces the same ID (idempotent ingestion).
    """

    @property
    def supported_content_type(self) -> str:
        return "text/plain"

    def parse(self, text: str, source: str) -> Document:
        document_id = _stable_id(source, length=16)
        filename = PurePath(source).name
        return Document(
            document_id=document_id,
            filename=filename,
            source_path=source,
            content_type=self.supported_content_type,
            text=text,
            metadata=dict(self.extra_metadata),
        )
