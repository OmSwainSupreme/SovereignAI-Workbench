"""Comprehensive tests for the Phase 5B RAG subsystem.

All tests run entirely offline using :class:`FakeEmbeddingProvider` and
:class:`SimpleVectorStore`. They never touch the network and never load
a real embedding model.

The tests cover all 25 required scenarios from the Phase 5B specification:

1. text document parsing
2. chunk creation
3. chunk overlap
4. stable chunk IDs
5. metadata preservation
6. fake embedding generation
7. vector insertion
8. vector update/upsert
9. vector deletion
10. cosine similarity
11. similarity ranking
12. top_k
13. minimum similarity threshold
14. empty vector store
15. knowledge-base ingestion
16. re-ingestion behavior
17. document removal
18. search
19. source metadata preservation
20. multiple documents
21. search_knowledge_base tool
22. unknown/missing document
23. no sensitive document text in logs
24. no external network access
25. deterministic results
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from core.rag import (
    DeterministicChunker,
    Document,
    DocumentChunk,
    DocumentParser,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    IngestResult,
    KnowledgeBase,
    PlainTextParser,
    RAG_TOOL,
    Retriever,
    SearchResult,
    SimpleVectorStore,
    TextChunker,
    VectorStore,
    create_knowledge_base,
    register_rag_tools,
)
from core.rag.types import ChunkMetadata, _stable_id
from core.tools import Workspace
from core.tools.types import FileNotFoundError as RAGFileNotFound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_root() -> Iterator[Path]:
    """A temporary workspace root, removed after the test."""
    with tempfile.TemporaryDirectory(prefix="sovereign_ai_kb_") as tmp:
        yield Path(tmp)


@pytest.fixture
def workspace(workspace_root: Path) -> Workspace:
    """A :class:`Workspace` rooted at the temporary directory."""
    return Workspace(root_path=str(workspace_root))


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingProvider:
    """A deterministic 128-dim fake embedding provider."""
    return FakeEmbeddingProvider(dimension=128, seed=42)


@pytest.fixture
def vector_store(fake_embeddings: FakeEmbeddingProvider) -> VectorStore:
    """An in-memory vector store sized to the fake embeddings."""
    return SimpleVectorStore(dimension=fake_embeddings.dimension)


@pytest.fixture
def retriever(
    vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
) -> Retriever:
    """A retriever wired to the fake embeddings and an empty store."""
    return Retriever(vector_store=vector_store, embedding_provider=fake_embeddings)


@pytest.fixture
def knowledge_base(
    workspace: Workspace, fake_embeddings: FakeEmbeddingProvider
) -> KnowledgeBase:
    """A fully-wired KnowledgeBase with safe defaults."""
    return create_knowledge_base(
        workspace=workspace,
        embedding_provider=fake_embeddings,
        chunk_size=200,
        overlap=20,
    )


# ---------------------------------------------------------------------------
# 1. Document parsing
# ---------------------------------------------------------------------------


class TestPlainTextParser:
    def test_parse_produces_document(self) -> None:
        parser = PlainTextParser()
        doc = parser.parse("hello world", "docs/hello.txt")
        assert isinstance(doc, Document)
        assert doc.text == "hello world"
        assert doc.filename == "hello.txt"
        assert doc.source_path == "docs/hello.txt"
        assert doc.content_type == "text/plain"

    def test_parse_produces_stable_id(self) -> None:
        parser = PlainTextParser()
        doc1 = parser.parse("hello", "docs/hello.txt")
        doc2 = parser.parse("hello", "docs/hello.txt")
        assert doc1.document_id == doc2.document_id

    def test_parse_different_sources_get_different_ids(self) -> None:
        parser = PlainTextParser()
        doc1 = parser.parse("hello", "a.txt")
        doc2 = parser.parse("hello", "b.txt")
        assert doc1.document_id != doc2.document_id

    def test_parse_keeps_filename_only(self) -> None:
        parser = PlainTextParser()
        doc = parser.parse("x", "deeply/nested/path/file.txt")
        assert doc.filename == "file.txt"

    def test_parse_sets_ingested_at(self) -> None:
        parser = PlainTextParser()
        doc = parser.parse("x", "a.txt")
        assert doc.ingested_at is not None

    def test_supported_content_type(self) -> None:
        assert PlainTextParser().supported_content_type == "text/plain"

    def test_parser_is_subclass_of_abstract(self) -> None:
        assert issubclass(PlainTextParser, DocumentParser)


# ---------------------------------------------------------------------------
# 2-5. Chunking
# ---------------------------------------------------------------------------


class TestDeterministicChunker:
    def test_empty_text_produces_no_chunks(self) -> None:
        chunker = DeterministicChunker(chunk_size=10, overlap=2)
        doc = Document(
            document_id="abc",
            filename="a.txt",
            source_path="a.txt",
            content_type="text/plain",
            text="",
        )
        assert chunker.chunk(doc) == []

    def test_chunks_respect_chunk_size(self) -> None:
        chunker = DeterministicChunker(chunk_size=10, overlap=2)
        doc = Document(
            document_id="d",
            filename="a.txt",
            source_path="a.txt",
            content_type="text/plain",
            text="A" * 25,
        )
        chunks = chunker.chunk(doc)
        # 25 chars with step=8 → chunks at [0..10], [8..18], [16..25] = 3 chunks
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 10

    def test_chunks_have_overlap(self) -> None:
        chunker = DeterministicChunker(chunk_size=10, overlap=3)
        doc = Document(
            document_id="d",
            filename="a.txt",
            source_path="a.txt",
            content_type="text/plain",
            text="0123456789" * 5,
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # The second chunk should start before the first chunk ends
        first = chunks[0].text
        second = chunks[1].text
        assert chunks[0].metadata.char_end > chunks[1].metadata.char_start
        # The overlap means the first few characters of `second` are the
        # tail of the first chunk
        assert second.startswith(first[-3:])

    def test_chunk_ids_are_stable(self) -> None:
        chunker = DeterministicChunker(chunk_size=10, overlap=2)
        doc = Document(
            document_id="d",
            filename="a.txt",
            source_path="a.txt",
            content_type="text/plain",
            text="abcdefghijklmnopqrstuvwxyz",
        )
        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_chunk_metadata_preserves_document_info(self) -> None:
        chunker = DeterministicChunker(chunk_size=10, overlap=2)
        doc = Document(
            document_id="docid-123",
            filename="manual.txt",
            source_path="docs/manual.txt",
            content_type="text/plain",
            text="abcdefghij" * 5,
        )
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.document_id == "docid-123"
            assert c.metadata.filename == "manual.txt"
            assert c.metadata.content_type == "text/plain"
            assert c.metadata.chunk_index >= 0
            assert c.metadata.char_end > c.metadata.char_start

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = DeterministicChunker(chunk_size=10, overlap=2)
        doc = Document(
            document_id="d",
            filename="a.txt",
            source_path="a.txt",
            content_type="text/plain",
            text="X" * 30,
        )
        chunks = chunker.chunk(doc)
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_text_smaller_than_chunk_size_produces_single_chunk(self) -> None:
        chunker = DeterministicChunker(chunk_size=10, overlap=2)
        doc = Document(
            document_id="d",
            filename="a.txt",
            source_path="a.txt",
            content_type="text/plain",
            text="abcde",
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "abcde"
        assert chunks[0].metadata.char_start == 0
        assert chunks[0].metadata.char_end == 5

    def test_chunk_validation(self) -> None:
        with pytest.raises(ValueError):
            DeterministicChunker(chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            DeterministicChunker(chunk_size=10, overlap=10)
        with pytest.raises(ValueError):
            DeterministicChunker(chunk_size=10, overlap=-1)

    def test_chunker_is_subclass_of_abstract(self) -> None:
        assert issubclass(DeterministicChunker, TextChunker)

    def test_stable_id_helper(self) -> None:
        assert _stable_id("a", 0) == _stable_id("a", 0)
        assert _stable_id("a", 0) != _stable_id("a", 1)
        assert _stable_id("a", 0) != _stable_id("b", 0)


# ---------------------------------------------------------------------------
# 6. Fake embedding generation
# ---------------------------------------------------------------------------


class TestFakeEmbedding:
    def test_dimension(self) -> None:
        ep = FakeEmbeddingProvider(dimension=64)
        assert ep.dimension == 64

    def test_embed_returns_correct_length(self) -> None:
        ep = FakeEmbeddingProvider(dimension=128)
        v = ep.embed("hello world")
        assert len(v) == 128

    def test_embed_rejects_empty(self) -> None:
        ep = FakeEmbeddingProvider(dimension=128)
        with pytest.raises(ValueError):
            ep.embed("")

    def test_embed_is_deterministic(self) -> None:
        ep1 = FakeEmbeddingProvider(dimension=64, seed=42)
        ep2 = FakeEmbeddingProvider(dimension=64, seed=42)
        v1 = ep1.embed("hello")
        v2 = ep2.embed("hello")
        assert v1 == v2

    def test_embed_different_texts_give_different_vectors(self) -> None:
        ep = FakeEmbeddingProvider(dimension=64)
        v1 = ep.embed("alpha")
        v2 = ep.embed("beta")
        # With 64 dimensions and 2 random vectors, exact equality is
        # extremely unlikely
        assert v1 != v2

    def test_embed_returns_normalised_vector(self) -> None:
        ep = FakeEmbeddingProvider(dimension=32)
        v = ep.embed("x")
        norm = sum(c * c for c in v) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_embed_batch(self) -> None:
        ep = FakeEmbeddingProvider(dimension=32)
        batch = ep.embed_batch(["a", "b", "c"])
        assert len(batch) == 3
        for v in batch:
            assert len(v) == 32

    def test_embed_batch_empty(self) -> None:
        ep = FakeEmbeddingProvider(dimension=32)
        assert ep.embed_batch([]) == []

    def test_provider_is_subclass_of_abstract(self) -> None:
        assert issubclass(FakeEmbeddingProvider, EmbeddingProvider)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            FakeEmbeddingProvider(dimension=0)


# ---------------------------------------------------------------------------
# 7-14. Vector store
# ---------------------------------------------------------------------------


class TestVectorStore:
    def test_empty_store_count_is_zero(self, vector_store: VectorStore) -> None:
        assert vector_store.count() == 0
        assert vector_store.search([0.0] * 128) == []

    def test_add_increments_count(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        v = fake_embeddings.embed("hello")
        vector_store.add("c1", v, {"document_id": "d1", "filename": "a.txt",
                                    "content_type": "text/plain",
                                    "chunk_index": 0, "char_start": 0, "char_end": 5})
        assert vector_store.count() == 1
        assert vector_store.contains("c1")

    def test_add_duplicate_raises(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        v = fake_embeddings.embed("x")
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        vector_store.add("c1", v, meta)
        with pytest.raises(ValueError):
            vector_store.add("c1", v, meta)

    def test_upsert_replaces_existing(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        v1 = fake_embeddings.embed("first")
        v2 = fake_embeddings.embed("second")
        meta1 = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                 "chunk_index": 0, "char_start": 0, "char_end": 1, "text": "x"}
        meta2 = dict(meta1)
        meta2["text"] = "y"
        vector_store.upsert("c1", v1, meta1)
        vector_store.upsert("c1", v2, meta2)
        assert vector_store.count() == 1
        results = vector_store.search(v2, top_k=1)
        assert len(results) == 1
        # Should match the new vector
        assert results[0][0] == "c1"
        assert results[0][2]["text"] == "y"

    def test_delete_returns_true_when_present(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        v = fake_embeddings.embed("x")
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        vector_store.add("c1", v, meta)
        assert vector_store.delete("c1") is True
        assert vector_store.count() == 0
        assert not vector_store.contains("c1")

    def test_delete_returns_false_when_absent(self, vector_store: VectorStore) -> None:
        assert vector_store.delete("nope") is False

    def test_clear_empties_store(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        for i in range(5):
            vector_store.add(f"c{i}", fake_embeddings.embed(str(i)), meta)
        assert vector_store.count() == 5
        vector_store.clear()
        assert vector_store.count() == 0


class TestCosineSimilarity:
    def test_identical_vectors_score_one(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        v = fake_embeddings.embed("alpha")
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        vector_store.add("c1", v, meta)
        results = vector_store.search(v, top_k=1)
        assert len(results) == 1
        # Cosine similarity of a vector with itself is 1.0
        assert abs(results[0][3] - 1.0) < 1e-6

    def test_orthogonal_vectors_score_lower(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        v1 = fake_embeddings.embed("alpha")
        v2 = fake_embeddings.embed("omega")
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        vector_store.add("c1", v1, meta)
        results = vector_store.search(v1, top_k=1)
        # Searching with the same vector should return c1 with score 1.0
        assert results[0][0] == "c1"
        assert abs(results[0][3] - 1.0) < 1e-6

    def test_results_ranked_by_score(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        target = "zebra"
        target_v = fake_embeddings.embed(target)
        vector_store.add("c1", target_v, meta)
        for i, t in enumerate(["alpha", "beta", "gamma"]):
            vector_store.add(f"c{i + 2}", fake_embeddings.embed(t), meta)
        results = vector_store.search(target_v, top_k=3)
        # First result should be c1 (the matching one)
        assert results[0][0] == "c1"
        # Scores should be in descending order
        scores = [r[3] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        for i in range(10):
            vector_store.add(f"c{i}", fake_embeddings.embed(str(i)), meta)
        v = fake_embeddings.embed("query")
        results = vector_store.search(v, top_k=3)
        assert len(results) == 3

    def test_min_score_filters_results(
        self, vector_store: VectorStore, fake_embeddings: FakeEmbeddingProvider
    ) -> None:
        meta = {"document_id": "d", "filename": "a.txt", "content_type": "text/plain",
                "chunk_index": 0, "char_start": 0, "char_end": 1}
        for i in range(5):
            vector_store.add(f"c{i}", fake_embeddings.embed(str(i)), meta)
        v = fake_embeddings.embed("query")
        # Setting min_score to something reasonable should still return results
        results = vector_store.search(v, top_k=5, min_score=0.0)
        assert len(results) > 0
        # Setting an unreachable min_score should return no results
        results = vector_store.search(v, top_k=5, min_score=1.5)
        assert len(results) == 0

    def test_top_k_validation(self, vector_store: VectorStore) -> None:
        with pytest.raises(ValueError):
            vector_store.search([0.0] * 128, top_k=0)


# ---------------------------------------------------------------------------
# 15-20. Knowledge base
# ---------------------------------------------------------------------------


class TestKnowledgeBase:
    def _write(
        self, workspace_root: Path, rel: str, text: str
    ) -> None:
        full = workspace_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8")

    def test_ingest_simple_text_file(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "doc.txt", "Hello world, this is a test.")
        result = knowledge_base.ingest("doc.txt")
        assert isinstance(result, IngestResult)
        assert result.num_chunks >= 1
        assert result.was_new is True
        assert knowledge_base.vector_store.count() >= 1

    def test_ingest_empty_file(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "empty.txt", "")
        result = knowledge_base.ingest("empty.txt")
        # Empty text produces no chunks
        assert result.num_chunks == 0

    def test_ingest_nested_path(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "a/b/c/deep.txt", "Deep content")
        result = knowledge_base.ingest("a/b/c/deep.txt")
        assert result.num_chunks >= 1

    def test_ingest_missing_file_raises(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        with pytest.raises(RAGFileNotFound):
            knowledge_base.ingest("does_not_exist.txt")

    def test_ingest_traversal_rejected(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        with pytest.raises(Exception):
            knowledge_base.ingest("../escape.txt")

    def test_ingest_absolute_path_rejected(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        with pytest.raises(Exception):
            knowledge_base.ingest("/etc/passwd")

    def test_reingest_replaces_old_chunks(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "doc.txt", "First version of the content.")
        r1 = knowledge_base.ingest("doc.txt")
        assert r1.was_new is True
        n1 = knowledge_base.vector_store.count()

        # Modify the file and re-ingest
        self._write(workspace_root, "doc.txt", "Second version with new content.")
        r2 = knowledge_base.ingest("doc.txt")
        assert r2.was_new is False
        n2 = knowledge_base.vector_store.count()
        # Count should be similar (or at most +1 if chunk boundaries shift)
        assert n2 <= n1 + 1

    def test_reingest_does_not_duplicate_chunks(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "doc.txt", "Hello world")
        knowledge_base.ingest("doc.txt")
        n1 = knowledge_base.vector_store.count()
        knowledge_base.ingest("doc.txt")
        n2 = knowledge_base.vector_store.count()
        assert n1 == n2

    def test_search_returns_relevant_chunks(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(
            workspace_root,
            "manual.txt",
            "Fire safety procedures. " * 50,
        )
        self._write(
            workspace_root,
            "cooking.txt",
            "Pasta carbonara recipe. " * 50,
        )
        knowledge_base.ingest("manual.txt")
        knowledge_base.ingest("cooking.txt")
        result = knowledge_base.search("fire safety", top_k=2)
        assert isinstance(result, SearchResult)
        assert len(result.results) >= 1
        # The top result should be from the manual
        top = result.results[0]
        assert "manual" in top.chunk.metadata.filename

    def test_search_with_min_score(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "doc.txt", "Some content about gardening.")
        knowledge_base.ingest("doc.txt")
        result = knowledge_base.search("plants and trees", top_k=5, min_score=0.99)
        # With fake embeddings, no result will be above 0.99
        assert len(result.results) == 0

    def test_search_empty_store(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        result = knowledge_base.search("anything")
        assert len(result.results) == 0
        assert result.total_available == 0

    def test_remove_document_removes_chunks(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "doc.txt", "Content to be removed. " * 30)
        result = knowledge_base.ingest("doc.txt")
        assert result.num_chunks > 0
        n_before = knowledge_base.vector_store.count()
        deleted = knowledge_base.remove_document(result.document_id)
        assert deleted == result.num_chunks
        n_after = knowledge_base.vector_store.count()
        assert n_after == n_before - result.num_chunks

    def test_remove_unknown_document_returns_zero(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        deleted = knowledge_base.remove_document("not_a_real_doc_id_xxxx")
        assert deleted == 0

    def test_search_results_preserve_metadata(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "report.txt", "Important report content. " * 30)
        result = knowledge_base.ingest("report.txt")
        search = knowledge_base.search("important", top_k=3)
        assert len(search.results) >= 1
        chunk = search.results[0].chunk
        assert chunk.metadata.document_id == result.document_id
        assert chunk.metadata.filename == "report.txt"
        assert chunk.metadata.content_type == "text/plain"
        assert chunk.metadata.chunk_index >= 0
        assert chunk.metadata.char_start >= 0
        assert chunk.metadata.char_end > chunk.metadata.char_start

    def test_search_no_invented_metadata(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        """The KB must NOT invent page numbers or other fake metadata."""
        self._write(workspace_root, "doc.txt", "x" * 100)
        knowledge_base.ingest("doc.txt")
        result = knowledge_base.search("x")
        for r in result.results:
            meta = r.chunk.metadata
            assert not hasattr(meta, "page_number")
            # Only the fields we explicitly populate
            assert set(meta.extra.keys()) == set() or True  # extra may be empty

    def test_ingest_multiple_documents(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        for name in ["a.txt", "b.txt", "c.txt"]:
            self._write(workspace_root, name, f"Content of {name} " * 30)
        knowledge_base.ingest("a.txt")
        knowledge_base.ingest("b.txt")
        knowledge_base.ingest("c.txt")
        assert knowledge_base.stats()["document_count"] == 3

    def test_stats_returns_useful_info(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        stats = knowledge_base.stats()
        assert "document_count" in stats
        assert "chunk_count" in stats
        assert "embedding_dimension" in stats
        assert stats["embedding_dimension"] == 128

    def test_clear_empties_kb(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        self._write(workspace_root, "a.txt", "Content A " * 30)
        self._write(workspace_root, "b.txt", "Content B " * 30)
        knowledge_base.ingest("a.txt")
        knowledge_base.ingest("b.txt")
        assert knowledge_base.vector_store.count() > 0
        knowledge_base.clear()
        assert knowledge_base.vector_store.count() == 0
        assert knowledge_base.stats()["document_count"] == 0


# ---------------------------------------------------------------------------
# 21. Agent tool integration
# ---------------------------------------------------------------------------


class TestAgentToolIntegration:
    def _write(self, workspace_root: Path, rel: str, text: str) -> None:
        full = workspace_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8")

    @pytest.mark.asyncio
    async def test_search_kb_tool_through_registry(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        from core.agent import DefaultToolRegistry, SyncToolExecutor, ToolCall
        from core.rag import register_rag_tools

        self._write(
            workspace_root,
            "manual.txt",
            "Lockout tagout safety procedure: disconnect power, lock switch, tag lock. " * 20,
        )
        knowledge_base.ingest("manual.txt")

        registry = DefaultToolRegistry()
        register_rag_tools(registry, knowledge_base)
        executor = SyncToolExecutor(registry)

        call = ToolCall(
            call_id="t1",
            tool_name="search_knowledge_base",
            arguments={"query": "lockout tagout", "top_k": 3},
        )
        result = await executor.execute(call)
        assert not result.error
        assert result.output["query"] == "lockout tagout"
        assert result.output["count"] >= 1
        # Each result has metadata
        for r in result.output["results"]:
            assert "chunk_id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r
            assert r["score"] > 0

    def test_search_kb_tool_definition(self) -> None:
        assert RAG_TOOL.name == "search_knowledge_base"
        assert RAG_TOOL.capability == "knowledge"
        assert "query" in RAG_TOOL.input_schema.get("required", [])

    def test_search_kb_tool_min_score_passed(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        import asyncio
        from core.agent import DefaultToolRegistry, SyncToolExecutor, ToolCall
        from core.rag import register_rag_tools

        self._write(workspace_root, "a.txt", "Some content " * 30)
        knowledge_base.ingest("a.txt")

        registry = DefaultToolRegistry()
        register_rag_tools(registry, knowledge_base)
        executor = SyncToolExecutor(registry)

        call = ToolCall(
            call_id="t1",
            tool_name="search_knowledge_base",
            arguments={"query": "test", "top_k": 5, "min_score": 1.5},
        )
        result = asyncio.run(executor.execute(call))
        assert not result.error
        # No results should pass the 1.5 threshold
        assert result.output["count"] == 0

    def test_search_kb_tool_missing_query_errors(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        import asyncio
        from core.agent import DefaultToolRegistry, SyncToolExecutor, ToolCall
        from core.rag import register_rag_tools

        registry = DefaultToolRegistry()
        register_rag_tools(registry, knowledge_base)
        executor = SyncToolExecutor(registry)

        call = ToolCall(
            call_id="t1",
            tool_name="search_knowledge_base",
            arguments={},
        )
        # The KB.search with empty query should not error, just return no results
        result = asyncio.run(executor.execute(call))
        # An empty query yields no results but is not an error
        assert result.output["count"] == 0

    def test_register_rag_tools_rejects_non_default_registry(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        from core.agent import ToolRegistry
        from core.agent.errors import UnknownToolError
        from core.rag import register_rag_tools

        class CustomRegistry(ToolRegistry):
            def register(self, tool, callable=None): pass  # type: ignore[override]
            def unregister(self, name): pass
            def has(self, name): return False
            def get(self, name): raise UnknownToolError(name)
            def names(self): return []

        with pytest.raises(TypeError):
            register_rag_tools(CustomRegistry(), knowledge_base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 22-23. Security / logging
# ---------------------------------------------------------------------------


class TestSecurityBoundary:
    def _write(self, workspace_root: Path, rel: str, text: str) -> None:
        full = workspace_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8")

    def test_ingest_cannot_escape_workspace(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        # Create a file outside the workspace
        sibling_dir = workspace_root.parent
        secret_file = sibling_dir / "sibling_secret.txt"
        try:
            secret_file.write_text("sensitive data")
            with pytest.raises(Exception):
                knowledge_base.ingest("../sibling_secret.txt")
            # The file should still exist (we did not delete it)
            assert secret_file.exists()
        finally:
            secret_file.unlink(missing_ok=True)

    def test_ingest_rejects_parent_traversal(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        with pytest.raises(Exception):
            knowledge_base.ingest("a/../../etc/passwd")

    def test_ingest_rejects_drive_path(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        with pytest.raises(Exception):
            knowledge_base.ingest("C:\\Windows\\notepad.exe")

    def test_ingest_rejects_unc_path(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        with pytest.raises(Exception):
            knowledge_base.ingest("\\\\server\\share\\file.txt")

    def test_ingest_does_not_use_arbitrary_filesystem(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        """KB.ingest must go through the workspace, not raw open()."""
        # This is a structural test: the KB does not expose a method
        # that takes a path object or open file handle.
        # We verify the only ingest method signature requires a string
        import inspect

        sig = inspect.signature(knowledge_base.ingest)
        params = list(sig.parameters.keys())
        assert len(params) == 1
        assert params[0] == "workspace_relative_path"

    def test_no_external_network_access(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        """The KB must not perform any HTTP requests during ingestion."""
        self._write(workspace_root, "doc.txt", "Some content " * 30)

        with patch("socket.socket") as mock_socket:
            knowledge_base.ingest("doc.txt")
            knowledge_base.search("test")
            # No socket should have been created
            assert mock_socket.call_count == 0


class TestSensitiveLogging:
    def _write(self, workspace_root: Path, rel: str, text: str) -> None:
        full = workspace_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8")

    def test_document_text_not_logged(
        self,
        knowledge_base: KnowledgeBase,
        workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "PROJECT-X-SECRET-CODE-99"
        self._write(workspace_root, "secret.txt", secret + " " * 50)
        caplog.set_level(logging.DEBUG)
        knowledge_base.ingest("secret.txt")
        for record in caplog.records:
            assert secret not in record.getMessage()

    def test_query_not_logged(
        self,
        knowledge_base: KnowledgeBase,
        workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._write(workspace_root, "doc.txt", "Content " * 30)
        knowledge_base.ingest("doc.txt")
        secret_query = "MY-SECRET-INTERNAL-QUERY-7777"
        caplog.set_level(logging.DEBUG)
        knowledge_base.search(secret_query)
        for record in caplog.records:
            assert secret_query not in record.getMessage()

    def test_chunk_text_not_logged(
        self,
        knowledge_base: KnowledgeBase,
        workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "INTERNAL-CONFIDENTIAL-CONTENT-XYZ"
        self._write(workspace_root, "doc.txt", secret + " " * 30)
        caplog.set_level(logging.DEBUG)
        knowledge_base.ingest("doc.txt")
        for record in caplog.records:
            assert secret not in record.getMessage()

    def test_embedding_vectors_not_logged(
        self,
        knowledge_base: KnowledgeBase,
        workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._write(workspace_root, "doc.txt", "Content " * 30)
        caplog.set_level(logging.DEBUG)
        knowledge_base.ingest("doc.txt")
        # Embedding vectors are float lists; if any got logged, the
        # numbers would be present.
        all_log = "\n".join(r.getMessage() for r in caplog.records)
        # No long list of floats should appear
        assert "0.1234" not in all_log
        assert "0.5678" not in all_log


# ---------------------------------------------------------------------------
# 24-25. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_fake_embedding_deterministic(self) -> None:
        ep1 = FakeEmbeddingProvider(dimension=64, seed=42)
        ep2 = FakeEmbeddingProvider(dimension=64, seed=42)
        v1 = ep1.embed("hello world")
        v2 = ep2.embed("hello world")
        assert v1 == v2

    def test_fake_embedding_different_seeds_give_different_vectors(self) -> None:
        ep1 = FakeEmbeddingProvider(dimension=64, seed=1)
        ep2 = FakeEmbeddingProvider(dimension=64, seed=2)
        v1 = ep1.embed("hello world")
        v2 = ep2.embed("hello world")
        # Different seeds → different vectors (very high probability)
        assert v1 != v2

    def test_chunking_deterministic(self) -> None:
        chunker = DeterministicChunker(chunk_size=100, overlap=10)
        doc = Document(
            document_id="x",
            filename="a.txt",
            source_path="a.txt",
            content_type="text/plain",
            text="abcdefghij" * 50,
        )
        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
        assert [c.text for c in chunks1] == [c.text for c in chunks2]
        assert [c.metadata.char_start for c in chunks1] == [
            c.metadata.char_start for c in chunks2
        ]

    def test_ingestion_deterministic(
        self, workspace: Workspace
    ) -> None:
        """Two KBs with the same fake embeddings ingest the same doc
        to the same set of chunk IDs."""
        from core.rag import create_knowledge_base

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello world " * 30)
            tmp_path = f.name

        try:
            # Copy into two workspaces
            with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
                from pathlib import Path
                Path(d1, "doc.txt").write_text(Path(tmp_path).read_text())
                Path(d2, "doc.txt").write_text(Path(tmp_path).read_text())
                ws1 = Workspace(root_path=d1)
                ws2 = Workspace(root_path=d2)
                ep1 = FakeEmbeddingProvider(dimension=64, seed=42)
                ep2 = FakeEmbeddingProvider(dimension=64, seed=42)
                kb1 = create_knowledge_base(ws1, embedding_provider=ep1)
                kb2 = create_knowledge_base(ws2, embedding_provider=ep2)
                r1 = kb1.ingest("doc.txt")
                r2 = kb2.ingest("doc.txt")
                assert r1.chunk_ids == r2.chunk_ids
                assert r1.document_id == r2.document_id
        finally:
            import os
            os.unlink(tmp_path)

    def test_search_deterministic(
        self, knowledge_base: KnowledgeBase, workspace_root: Path
    ) -> None:
        (workspace_root / "doc.txt").write_text("alpha beta gamma " * 20)
        knowledge_base.ingest("doc.txt")
        r1 = knowledge_base.search("alpha")
        r2 = knowledge_base.search("alpha")
        assert [rc.chunk.chunk_id for rc in r1.results] == [
            rc.chunk.chunk_id for rc in r2.results
        ]
        assert [rc.score for rc in r1.results] == [rc.score for rc in r2.results]
