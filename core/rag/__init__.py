"""SovereignAI Workbench — Local RAG / Knowledge Base (Phase 5B).

Public surface for the RAG subsystem. All imports are from this module;
sub-modules are implementation details.

Example::

    from core.rag import (
        create_knowledge_base,
        FakeEmbeddingProvider,
        Workspace,
    )
    from core.rag import register_rag_tools, DefaultToolRegistry

    workspace = Workspace(root_path="/data/kb_workspace")
    kb = create_knowledge_base(workspace)

    # Ingest a document
    result = kb.ingest("manuals/safety-procedures.txt")

    # Search
    results = kb.search("fire safety steps")
    for r in results.results:
        print(f"[{r.score:.2f}] {r.chunk.metadata.filename}: {r.chunk.text[:80]}")

    # Agent integration
    registry = DefaultToolRegistry()
    register_rag_tools(registry, kb)
"""

from core.rag.config import RAGConfig, load_rag_config, load_rag_config_from_mapping
from core.rag.errors import RAGConfigurationError, RAGError, RAGIngestError, RAGSearchError
from core.rag.types import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    IngestResult,
    RetrievedChunk,
    SearchResult,
)

from core.rag.document_parser import DocumentParser, PlainTextParser
from core.rag.chunker import DeterministicChunker, TextChunker
from core.rag.embedding import EmbeddingProvider, FakeEmbeddingProvider
from core.rag.vector_store import SimpleVectorStore, VectorStore
from core.rag.retriever import Retriever
from core.rag.knowledge_base import KnowledgeBase, create_knowledge_base
from core.rag.tools import register_rag_tools, SEARCH_KB_TOOL as RAG_TOOL

__all__ = [
    # Types
    "Document",
    "DocumentChunk",
    "ChunkMetadata",
    "RetrievedChunk",
    "SearchResult",
    "IngestResult",
    # Errors
    "RAGError",
    "RAGConfigurationError",
    "RAGIngestError",
    "RAGSearchError",
    # Abstractions
    "DocumentParser",
    "TextChunker",
    "EmbeddingProvider",
    "VectorStore",
    "Retriever",
    "KnowledgeBase",
    # Concrete implementations
    "PlainTextParser",
    "DeterministicChunker",
    "FakeEmbeddingProvider",
    "SimpleVectorStore",
    # Factories
    "create_knowledge_base",
    # Configuration
    "RAGConfig",
    "load_rag_config",
    "load_rag_config_from_mapping",
    # Agent tool
    "RAG_TOOL",
    "register_rag_tools",
]
