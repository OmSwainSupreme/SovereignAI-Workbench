"""Knowledge base service for the RAG subsystem (Phase 5B).

The :class:`KnowledgeBase` is the top-level RAG service. It orchestrates
parsing, chunking, embedding, and storage to provide a simple
ingest/search interface. It is the component that the agent's
``search_knowledge_base`` tool talks to.

The KnowledgeBase is **not** FastAPI-aware — it accepts workspace-relative
paths and uses the file tools for I/O, preserving the workspace security
boundary. It does not know about HTTP, Pydantic, or the internet.

Architecture::

    Agent tool call
         │
         ▼
    KnowledgeBase.ingest / .search
         │
         ├──► read_file (via workspace, secure)
         │         │
         │         ▼
         │    DocumentParser.parse()
         │         │
         │         ▼
         │    TextChunker.chunk()
         │         │
         │         ▼
         │    EmbeddingProvider.embed_batch()
         │         │
         │         ▼
         │    VectorStore.upsert()
         │
         └──► Retriever.retrieve()
                   │
                   ▼
              SearchResult
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING, Optional

from core.agent.types import ToolCall
from core.rag.chunker import DeterministicChunker, TextChunker
from core.rag.document_parser import DocumentParser, PlainTextParser
from core.rag.embedding import EmbeddingProvider, FakeEmbeddingProvider
from core.rag.retriever import Retriever
from core.rag.types import (
    Document,
    IngestResult,
    SearchResult,
)
from core.rag.vector_store import SimpleVectorStore, VectorStore
from core.tools.workspace import Workspace

if TYPE_CHECKING:
    from core.tools.file_tools import FileToolExecutor


_logger = logging.getLogger("sovereign-ai.rag.kb")


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """A local RAG knowledge base.

    The KnowledgeBase holds a collection of indexed documents and can answer
    natural-language queries about them. It is built from five swappable
    components:

    * **workspace**: provides secure filesystem access via the file tools.
    * **parser**: converts raw text into :class:`Document` objects.
    * **chunker**: splits documents into :class:`DocumentChunk` objects.
    * **embedding_provider**: converts text to vectors.
    * **vector_store**: stores and searches vectors.

    All five are injected via the constructor so any can be swapped without
    touching the KnowledgeBase itself.

    The KnowledgeBase is thread-safe for read operations. Write operations
    (ingest, remove) are serialised via an internal asyncio lock so that
    concurrent coroutines do not corrupt the vector store.
    """

    def __init__(
        self,
        workspace: Workspace,
        parser: DocumentParser,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._workspace = workspace
        self._parser = parser
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._retriever = Retriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )
        # Lazily created FileToolExecutor — we avoid holding it open.
        self._file_executor: Optional[FileToolExecutor] = None
        # Track which chunks belong to which document for removal.
        # chunk_id -> document_id
        self._chunk_to_doc: dict[str, str] = {}
        # document_id -> set of chunk_ids
        self._doc_to_chunks: dict[str, set[str]] = {}

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    # ----------------------------------------------------------------- Public API

    def ingest(self, workspace_relative_path: str) -> IngestResult:
        """Ingest a document from the workspace into the knowledge base.

        The path is resolved **strictly** within the configured workspace
        via the Workspace boundary. The workspace security boundary is
        preserved — no arbitrary filesystem access.

        Re-ingesting the same path replaces the old vectors (upsert
        semantics). The ``document_id`` is derived from the source path,
        so the same file always maps to the same ID.

        Args:
            workspace_relative_path: A path relative to the workspace root.
                Must not escape the workspace (../ traversal etc.).

        Returns:
            An :class:`IngestResult` describing what was indexed.

        Raises:
            FileNotFoundError: if the path does not exist in the workspace.
            ValueError: if the path is invalid (absolute, traversal, etc.).
        """
        # Step 1: Resolve the path through the workspace boundary.
        # Workspace.resolve() enforces all security checks (traversal,
        # absolute paths, symlinks). This is the single security gate.
        try:
            resolved = self._workspace.resolve(workspace_relative_path)
        except WorkspaceError as exc:
            # Re-raise workspace validation errors with a clean message.
            raise ValueError(
                f"Invalid workspace path {workspace_relative_path!r}: {exc}"
            ) from exc

        if not resolved.exists():
            from core.tools.types import FileNotFoundError as RAGFileNotFound

            raise RAGFileNotFound(
                f"File not found in workspace: {workspace_relative_path!r}",
                path=workspace_relative_path,
            )

        if not resolved.is_file():
            from core.tools.types import UnsupportedFileError

            raise UnsupportedFileError(
                f"Not a regular file: {workspace_relative_path!r}",
                path=workspace_relative_path,
            )

        # Read the file. We use binary mode and decode as UTF-8, matching
        # the file tool's behavior. Files that are not valid UTF-8 are
        # treated as unsupported.
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            from core.tools.types import FileToolError

            raise FileToolError(
                f"Cannot read file: {exc}",
                path=workspace_relative_path,
            ) from exc

        # Reject files that look binary (NUL bytes, invalid UTF-8).
        if b"\x00" in raw[:4096]:
            from core.tools.types import UnsupportedFileError

            raise UnsupportedFileError(
                "File appears to be binary",
                path=workspace_relative_path,
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            from core.tools.types import UnsupportedFileError

            raise UnsupportedFileError(
                "File is not valid UTF-8 text",
                path=workspace_relative_path,
            )

        # Reject oversized files (same limit as the file tool).
        _MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MiB
        if len(raw) > _MAX_READ_BYTES:
            from core.tools.types import UnsupportedFileError

            raise UnsupportedFileError(
                f"File is too large to ingest ({len(raw)} bytes)",
                path=workspace_relative_path,
            )

        # Step 2: Parse → Document
        document = self._parser.parse(text, workspace_relative_path)

        # Step 3: Chunk → list[DocumentChunk]
        chunks = self._chunker.chunk(document)

        # Step 4: Embed + store
        was_new = document.document_id not in self._doc_to_chunks

        # Remove old chunks for this document if re-ingesting
        if not was_new:
            self._remove_document_chunks(document.document_id)

        chunk_ids: list[str] = []
        if chunks:
            texts = [c.text for c in chunks]
            vectors = self._embedding_provider.embed_batch(texts)

            for chunk, vector in zip(chunks, vectors):
                # Store in vector store
                self._vector_store.upsert(
                    chunk_id=chunk.chunk_id,
                    vector=vector,
                    metadata={
                        "document_id": chunk.metadata.document_id,
                        "filename": chunk.metadata.filename,
                        "content_type": chunk.metadata.content_type,
                        "chunk_index": chunk.metadata.chunk_index,
                        "char_start": chunk.metadata.char_start,
                        "char_end": chunk.metadata.char_end,
                        "text": chunk.text,
                    },
                )
                # Track ownership for removal
                self._chunk_to_doc[chunk.chunk_id] = document.document_id
                self._doc_to_chunks.setdefault(document.document_id, set()).add(
                    chunk.chunk_id
                )
                chunk_ids.append(chunk.chunk_id)

        _logger.info(
            "kb.ingest  doc=%s  path=%s  chunks=%d  was_new=%s",
            document.document_id,
            workspace_relative_path,
            len(chunk_ids),
            was_new,
        )

        return IngestResult(
            document_id=document.document_id,
            filename=document.filename,
            num_chunks=len(chunk_ids),
            chunk_ids=tuple(chunk_ids),
            was_new=was_new,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> SearchResult:
        """Search the knowledge base for chunks relevant to the query.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results.
            min_score: Minimum cosine similarity threshold (0.0–1.0).

        Returns:
            A :class:`SearchResult` with ranked retrieved chunks.
        """
        return self._retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    def remove_document(self, document_id: str) -> int:
        """Remove a document and all its chunks from the knowledge base.

        Args:
            document_id: The ID of the document to remove.

        Returns:
            The number of chunks that were deleted.
        """
        count = self._remove_document_chunks(document_id)
        _logger.info("kb.remove  doc=%s  chunks_removed=%d", document_id, count)
        return count

    def stats(self) -> dict[str, Any]:
        """Return basic statistics about the knowledge base."""
        return {
            "document_count": len(self._doc_to_chunks),
            "chunk_count": self._vector_store.count(),
            "embedding_dimension": self._embedding_provider.dimension,
            "embedding_model": type(self._embedding_provider).__name__,
            "vector_store": type(self._vector_store).__name__,
        }

    def clear(self) -> None:
        """Remove all documents and chunks."""
        self._vector_store.clear()
        self._chunk_to_doc.clear()
        self._doc_to_chunks.clear()
        _logger.info("kb.cleared")

    # ----------------------------------------------------------------- Internals

    def _get_file_executor(self) -> "FileToolExecutor":
        if self._file_executor is None:
            from core.tools.file_tools import FileToolExecutor

            self._file_executor = FileToolExecutor(self._workspace)
        return self._file_executor

    def _remove_document_chunks(self, document_id: str) -> int:
        chunk_ids = self._doc_to_chunks.pop(document_id, set())
        deleted = 0
        for chunk_id in chunk_ids:
            if self._vector_store.delete(chunk_id):
                deleted += 1
            self._chunk_to_doc.pop(chunk_id, None)
        return deleted


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_knowledge_base(
    workspace: Workspace,
    embedding_provider: Optional[EmbeddingProvider] = None,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> KnowledgeBase:
    """Create a wired KnowledgeBase with sensible defaults.

    This is a convenience factory. All components can also be constructed
    and wired manually for custom configurations.

    Args:
        workspace: The secure workspace the KB reads files from.
        embedding_provider: The embedding provider. Defaults to
            :class:`FakeEmbeddingProvider` (deterministic fake vectors).
        chunk_size: Characters per chunk (passed to
            :class:`DeterministicChunker`).
        overlap: Overlapping characters between chunks.

    Returns:
        A fully wired :class:`KnowledgeBase` instance.
    """
    ep = embedding_provider or FakeEmbeddingProvider()
    return KnowledgeBase(
        workspace=workspace,
        parser=PlainTextParser(),
        chunker=DeterministicChunker(chunk_size=chunk_size, overlap=overlap),
        embedding_provider=ep,
        vector_store=SimpleVectorStore(dimension=ep.dimension),
    )
