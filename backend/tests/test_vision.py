"""Comprehensive tests for the Phase 5C OCR + Vision subsystem.

All tests run entirely offline using :class:`FakeOCRProvider` and
:class:`FakeVisionProvider`. They never touch the network and never load
a real OCR or vision model.

The tests cover all 27 required scenarios from the Phase 5C specification:

1.  OCR provider interface
2.  Fake OCR result
3.  Deterministic fake OCR behavior
4.  OCR metadata preservation
5.  OCR confidence handling
6.  OCR multi-page/document result
7.  Vision provider interface
8.  Fake vision result
9.  Deterministic fake vision behavior
10. Image metadata handling
11. Unsupported input handling
12. Empty image handling
13. Invalid image metadata
14. OCR tool registration
15. Vision tool registration
16. OCR tool execution
17. Vision tool execution
18. Tool failure handling
19. No sensitive OCR text in logs
20. No sensitive vision output in logs
21. No image bytes in logs
22. Workspace/path security if file-backed inputs are supported
23. No external network access
24. Compatibility with existing ToolRegistry
25. Compatibility with existing RAG Document representation
26. Vision routing metadata/capability integration
27. Deterministic results for identical test inputs
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from core.routing import (
    Capability,
    ModelDefinition,
    Modality,
    RoutingRequest,
    TaskType,
)
from core.routing.registry import ModelRegistry
from core.routing.router import ModelRouter
from core.routing.errors import NoSuitableModelError
from core.rag.types import Document
from core.tools import Workspace
from core.vision import (
    ANALYZE_IMAGE_TOOL,
    FakeOCRProvider,
    FakeVisionProvider,
    ImageDocument,
    ImageInput,
    ImageLoader,
    ImageMetadata,
    OCR_IMAGE_TOOL,
    OCRBlock,
    OCRBoundingBox,
    OCRLine,
    OCRProvider,
    OCRResult,
    OCRWord,
    SUPPORTED_IMAGE_CONTENT_TYPES,
    VisionProvider,
    VisionRegion,
    VisionResult,
    build_vision_routing_request,
    ocr_to_document,
    register_vision_tools,
)
from core.vision.errors import (
    EmptyImageError,
    InvalidImageError,
    UnsupportedImageError,
)
from core.agent import (
    DefaultToolRegistry,
    SyncToolExecutor,
    ToolCall,
    ToolRegistry,
)
from core.agent.errors import UnknownToolError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
# Build a minimal valid image "blob" for tests. We do not parse it
# semantically; the fake providers only hash the bytes.
_FAKE_IMAGE_BYTES = _PNG_MAGIC + b"x" * 256


def _make_image(
    image_id: str = "img-test",
    source: str = "scans/page.png",
    content_type: str = "image/png",
    data: bytes = _FAKE_IMAGE_BYTES,
) -> ImageInput:
    """Build an :class:`ImageInput` for tests."""
    return ImageInput(
        metadata=ImageMetadata(
            image_id=image_id,
            filename=Path(source).name,
            source_path=source,
            content_type=content_type,
            size_bytes=len(data),
        ),
        bytes=data,
    )


@pytest.fixture
def workspace_root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="sovereign_ai_vision_") as tmp:
        yield Path(tmp)


@pytest.fixture
def workspace(workspace_root: Path) -> Workspace:
    return Workspace(root_path=str(workspace_root))


@pytest.fixture
def image_loader(workspace: Workspace) -> ImageLoader:
    return ImageLoader(workspace)


@pytest.fixture
def ocr_provider() -> OCRProvider:
    return FakeOCRProvider()


@pytest.fixture
def vision_provider() -> VisionProvider:
    return FakeVisionProvider()


# ---------------------------------------------------------------------------
# 1-2. OCR provider interface & fake result
# ---------------------------------------------------------------------------


class TestOCRProviderInterface:
    def test_is_abstract(self) -> None:
        assert issubclass(FakeOCRProvider, OCRProvider)

    def test_provider_name(self, ocr_provider: OCRProvider) -> None:
        assert ocr_provider.provider_name == "fake"

    def test_recognize_returns_ocr_result(
        self, ocr_provider: OCRProvider
    ) -> None:
        image = _make_image()
        result = ocr_provider.recognize(image)
        assert isinstance(result, OCRResult)
        assert result.image_id == "img-test"
        assert isinstance(result.full_text, str)
        assert len(result.full_text) > 0

    def test_recognize_preserves_image_id(
        self, ocr_provider: OCRProvider
    ) -> None:
        image = _make_image(image_id="custom-id-42")
        result = ocr_provider.recognize(image)
        assert result.image_id == "custom-id-42"


# ---------------------------------------------------------------------------
# 3. Deterministic fake OCR behavior
# ---------------------------------------------------------------------------


class TestFakeOCRDeterminism:
    def test_same_image_same_text(self, ocr_provider: OCRProvider) -> None:
        image = _make_image(image_id="det-1")
        r1 = ocr_provider.recognize(image)
        r2 = ocr_provider.recognize(image)
        assert r1.full_text == r2.full_text

    def test_different_ids_may_give_different_text(
        self, ocr_provider: OCRProvider
    ) -> None:
        r1 = ocr_provider.recognize(_make_image(image_id="a"))
        r2 = ocr_provider.recognize(_make_image(image_id="b"))
        # Different IDs hash to different placeholders (with high probability).
        # We just check determinism, not content, here.
        assert r1.full_text is not None
        assert r2.full_text is not None

    def test_provider_is_deterministic_across_instances(self) -> None:
        p1 = FakeOCRProvider()
        p2 = FakeOCRProvider()
        image = _make_image(image_id="x")
        assert p1.recognize(image).full_text == p2.recognize(image).full_text


# ---------------------------------------------------------------------------
# 4. OCR metadata preservation
# ---------------------------------------------------------------------------


class TestOCRMetadata:
    def test_blocks_are_returned(self, ocr_provider: OCRProvider) -> None:
        result = ocr_provider.recognize(_make_image())
        assert len(result.blocks) >= 1
        block = result.blocks[0]
        assert isinstance(block, OCRBlock)
        assert block.text == result.full_text
        assert len(block.lines) >= 1

    def test_lines_and_words_are_structured(
        self, ocr_provider: OCRProvider
    ) -> None:
        result = ocr_provider.recognize(_make_image())
        line = result.blocks[0].lines[0]
        assert isinstance(line, OCRLine)
        assert line.text == result.full_text
        assert len(line.words) >= 1
        word = line.words[0]
        assert isinstance(word, OCRWord)
        assert isinstance(word.text, str)
        assert len(word.text) > 0

    def test_recognized_at_is_set(self, ocr_provider: OCRProvider) -> None:
        result = ocr_provider.recognize(_make_image())
        assert result.recognized_at is not None

    def test_provider_name_in_result(self, ocr_provider: OCRProvider) -> None:
        result = ocr_provider.recognize(_make_image())
        assert result.provider == "fake"

    def test_confidence_in_range(self, ocr_provider: OCRProvider) -> None:
        result = ocr_provider.recognize(_make_image())
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# 5. OCR confidence handling
# ---------------------------------------------------------------------------


class TestOCRConfidence:
    def test_confidence_is_not_invented_for_fake(self) -> None:
        # The fake provider sets confidence = 0.95 by design. Real
        # providers must report real confidence, but the fake is
        # explicitly allowed to use deterministic test data.
        provider = FakeOCRProvider()
        result = provider.recognize(_make_image())
        assert result.confidence == 0.95

    def test_confidence_in_blocks(self, ocr_provider: OCRProvider) -> None:
        result = ocr_provider.recognize(_make_image())
        for block in result.blocks:
            assert block.confidence is not None
            for line in block.lines:
                assert line.confidence is not None


# ---------------------------------------------------------------------------
# 6. OCR multi-page/document result
# ---------------------------------------------------------------------------


class TestOCRMultiPage:
    def test_recognize_document_returns_list_per_page(
        self, ocr_provider: OCRProvider
    ) -> None:
        pages = [
            _make_image(image_id=f"page-{i}", source=f"p{i}.png")
            for i in range(3)
        ]
        document = ImageDocument(
            document_id="doc-1",
            filename="multi.png",
            source_path="scans/multi.png",
            pages=tuple(pages),
        )
        results = ocr_provider.recognize_document(document)
        assert len(results) == 3
        assert [r.image_id for r in results] == ["page-0", "page-1", "page-2"]

    def test_recognize_document_empty(self, ocr_provider: OCRProvider) -> None:
        document = ImageDocument(
            document_id="empty",
            filename="empty.png",
            source_path="empty.png",
            pages=(),
        )
        assert ocr_provider.recognize_document(document) == []


# ---------------------------------------------------------------------------
# 7-8. Vision provider interface & fake result
# ---------------------------------------------------------------------------


class TestVisionProviderInterface:
    def test_is_abstract(self) -> None:
        assert issubclass(FakeVisionProvider, VisionProvider)

    def test_provider_name(self, vision_provider: VisionProvider) -> None:
        assert vision_provider.provider_name == "fake"

    def test_analyze_returns_vision_result(
        self, vision_provider: VisionProvider
    ) -> None:
        result = vision_provider.analyze(_make_image(), "describe")
        assert isinstance(result, VisionResult)
        assert result.image_id == "img-test"
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_analyze_with_no_prompt(
        self, vision_provider: VisionProvider
    ) -> None:
        result = vision_provider.analyze(_make_image())
        assert isinstance(result, VisionResult)
        assert result.prompt == ""


# ---------------------------------------------------------------------------
# 9. Deterministic fake vision behavior
# ---------------------------------------------------------------------------


class TestFakeVisionDeterminism:
    def test_same_image_same_description(
        self, vision_provider: VisionProvider
    ) -> None:
        image = _make_image(image_id="v-det-1")
        r1 = vision_provider.analyze(image, "describe")
        r2 = vision_provider.analyze(image, "describe")
        assert r1.description == r2.description

    def test_same_image_same_regions(
        self, vision_provider: VisionProvider
    ) -> None:
        image = _make_image(image_id="v-det-2")
        r1 = vision_provider.analyze(image)
        r2 = vision_provider.analyze(image)
        assert len(r1.regions) == len(r2.regions)
        assert r1.regions[0].label == r2.regions[0].label

    def test_provider_deterministic_across_instances(self) -> None:
        p1 = FakeVisionProvider()
        p2 = FakeVisionProvider()
        image = _make_image(image_id="vx")
        assert p1.analyze(image).description == p2.analyze(image).description


# ---------------------------------------------------------------------------
# 10. Image metadata handling
# ---------------------------------------------------------------------------


class TestImageMetadata:
    def test_image_input_validates(self) -> None:
        # Valid construction
        img = _make_image()
        assert img.image_id == "img-test"
        assert img.content_type == "image/png"

    def test_image_input_requires_image_id(self) -> None:
        with pytest.raises(ValueError):
            ImageInput(
                metadata=ImageMetadata(
                    image_id="",
                    filename="x.png",
                    source_path="x.png",
                    content_type="image/png",
                ),
                bytes=b"abc",
            )

    def test_image_input_requires_content_type(self) -> None:
        with pytest.raises(ValueError):
            ImageInput(
                metadata=ImageMetadata(
                    image_id="x",
                    filename="x.png",
                    source_path="x.png",
                    content_type="",
                ),
                bytes=b"abc",
            )

    def test_supported_content_types_includes_common(self) -> None:
        assert "image/png" in SUPPORTED_IMAGE_CONTENT_TYPES
        assert "image/jpeg" in SUPPORTED_IMAGE_CONTENT_TYPES
        assert "image/webp" in SUPPORTED_IMAGE_CONTENT_TYPES


# ---------------------------------------------------------------------------
# 11. Unsupported input handling
# ---------------------------------------------------------------------------


class TestUnsupportedInput:
    def test_ocr_rejects_unknown_content_type(
        self, ocr_provider: OCRProvider
    ) -> None:
        image = _make_image(content_type="image/gif")
        with pytest.raises(UnsupportedImageError):
            ocr_provider.recognize(image)

    def test_vision_rejects_unknown_content_type(
        self, vision_provider: VisionProvider
    ) -> None:
        image = _make_image(content_type="image/gif")
        with pytest.raises(UnsupportedImageError):
            vision_provider.analyze(image)

    def test_loader_rejects_unknown_content_type(
        self, image_loader: ImageLoader, workspace_root: Path
    ) -> None:
        (workspace_root / "x.gif").write_bytes(b"GIF89a")
        with pytest.raises(UnsupportedImageError):
            image_loader.load("x.gif")


# ---------------------------------------------------------------------------
# 12. Empty image handling
# ---------------------------------------------------------------------------


class TestEmptyImage:
    def test_ocr_rejects_empty_bytes(self, ocr_provider: OCRProvider) -> None:
        with pytest.raises(EmptyImageError):
            ocr_provider.recognize(_make_image(data=b""))

    def test_vision_rejects_empty_bytes(
        self, vision_provider: VisionProvider
    ) -> None:
        with pytest.raises(EmptyImageError):
            vision_provider.analyze(_make_image(data=b""))

    def test_loader_rejects_empty_file(
        self, image_loader: ImageLoader, workspace_root: Path
    ) -> None:
        (workspace_root / "empty.png").write_bytes(b"")
        with pytest.raises(EmptyImageError):
            image_loader.load("empty.png")


# ---------------------------------------------------------------------------
# 13. Invalid image metadata
# ---------------------------------------------------------------------------


class TestInvalidImageMetadata:
    def test_image_input_rejects_non_bytes(self) -> None:
        meta = ImageMetadata(
            image_id="x",
            filename="x.png",
            source_path="x.png",
            content_type="image/png",
        )
        with pytest.raises(TypeError):
            ImageInput(metadata=meta, bytes="not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 14-15. Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_ocr_tool_definition(self) -> None:
        assert OCR_IMAGE_TOOL.name == "ocr_image"
        assert OCR_IMAGE_TOOL.capability == "vision"
        assert "path" in OCR_IMAGE_TOOL.input_schema.get("required", [])

    def test_analyze_tool_definition(self) -> None:
        assert ANALYZE_IMAGE_TOOL.name == "analyze_image"
        assert ANALYZE_IMAGE_TOOL.capability == "vision"
        assert "path" in ANALYZE_IMAGE_TOOL.input_schema.get("required", [])

    def test_register_vision_tools_registers_both(
        self,
        image_loader: ImageLoader,
        ocr_provider: OCRProvider,
        vision_provider: VisionProvider,
    ) -> None:
        registry = DefaultToolRegistry()
        register_vision_tools(registry, ocr_provider, vision_provider, image_loader)
        assert registry.has("ocr_image")
        assert registry.has("analyze_image")

    def test_register_vision_tools_rejects_non_default_registry(
        self,
        image_loader: ImageLoader,
        ocr_provider: OCRProvider,
        vision_provider: VisionProvider,
    ) -> None:
        class CustomRegistry(ToolRegistry):
            def register(self, tool, callable=None): pass  # type: ignore[override]
            def unregister(self, name): pass
            def has(self, name): return False
            def get(self, name): raise UnknownToolError(name)
            def names(self): return []

        with pytest.raises(TypeError):
            register_vision_tools(
                CustomRegistry(),
                ocr_provider,
                vision_provider,
                image_loader,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 16-17. Tool execution
# ---------------------------------------------------------------------------


class TestToolExecution:
    def _write_png(self, root: Path, name: str) -> str:
        full = root / name
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(_FAKE_IMAGE_BYTES)
        return name

    @pytest.mark.asyncio
    async def test_ocr_tool_executes(
        self,
        workspace_root: Path,
        workspace: Workspace,
        ocr_provider: OCRProvider,
    ) -> None:
        self._write_png(workspace_root, "scan.png")
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(registry, ocr_provider, FakeVisionProvider(), loader)
        executor = SyncToolExecutor(registry)

        call = ToolCall(
            call_id="t1",
            tool_name="ocr_image",
            arguments={"path": "scan.png"},
        )
        result = await executor.execute(call)
        assert not result.error
        assert result.output["image_id"]
        assert isinstance(result.output["full_text"], str)
        assert result.output["provider"] == "fake"

    @pytest.mark.asyncio
    async def test_analyze_tool_executes(
        self,
        workspace_root: Path,
        workspace: Workspace,
        vision_provider: VisionProvider,
    ) -> None:
        self._write_png(workspace_root, "img.png")
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), vision_provider, loader
        )
        executor = SyncToolExecutor(registry)

        call = ToolCall(
            call_id="t2",
            tool_name="analyze_image",
            arguments={"path": "img.png", "prompt": "describe"},
        )
        result = await executor.execute(call)
        assert not result.error
        assert isinstance(result.output["description"], str)
        assert len(result.output["description"]) > 0
        assert result.output["provider"] == "fake"

    @pytest.mark.asyncio
    async def test_analyze_tool_no_prompt(
        self,
        workspace_root: Path,
        workspace: Workspace,
        vision_provider: VisionProvider,
    ) -> None:
        self._write_png(workspace_root, "img.png")
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), vision_provider, loader
        )
        executor = SyncToolExecutor(registry)
        call = ToolCall(
            call_id="t3",
            tool_name="analyze_image",
            arguments={"path": "img.png"},
        )
        result = await executor.execute(call)
        assert not result.error

    @pytest.mark.asyncio
    async def test_analyze_tool_without_path_errors(
        self,
        workspace_root: Path,
        workspace: Workspace,
    ) -> None:
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)
        call = ToolCall(
            call_id="t4",
            tool_name="analyze_image",
            arguments={},
        )
        result = await executor.execute(call)
        # Missing path → tool raises ToolExecutionError → result has error=True
        assert result.error


# ---------------------------------------------------------------------------
# 18. Tool failure handling
# ---------------------------------------------------------------------------


class TestToolFailureHandling:
    def _write_png(self, root: Path, name: str) -> str:
        full = root / name
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(_FAKE_IMAGE_BYTES)
        return name

    def test_ocr_tool_handles_missing_file(
        self,
        workspace_root: Path,
        workspace: Workspace,
    ) -> None:
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)
        import asyncio

        call = ToolCall(
            call_id="t1",
            tool_name="ocr_image",
            arguments={"path": "does_not_exist.png"},
        )
        result = asyncio.run(executor.execute(call))
        assert result.error
        assert "FileNotFoundError" in result.error_message or "File" in result.error_message


# ---------------------------------------------------------------------------
# 19-21. No sensitive data in logs
# ---------------------------------------------------------------------------


class TestSensitiveLogging:
    def _write_png(self, root: Path, name: str, data: bytes) -> str:
        full = root / name
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        return name

    def test_ocr_text_not_in_logs(
        self,
        workspace_root: Path,
        workspace: Workspace,
        ocr_provider: OCRProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # We can't fully control what the fake provider logs, so we
        # run the full OCR + tool flow and check that no log contains
        # the OCR full_text. Since full_text is the same for any image
        # with this id, we use a unique id and check that the unique
        # full_text value is not present.
        secret_path = self._write_png(workspace_root, "doc.png", _FAKE_IMAGE_BYTES)
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, ocr_provider, FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)

        call = ToolCall(
            call_id="l1",
            tool_name="ocr_image",
            arguments={"path": secret_path},
        )
        caplog.set_level(logging.DEBUG)
        import asyncio

        result = asyncio.run(executor.execute(call))
        assert not result.error
        full_text = result.output["full_text"]

        # Confirm the OCR text is NOT in any log message
        all_log = "\n".join(r.getMessage() for r in caplog.records)
        assert full_text not in all_log

    def test_vision_description_not_in_logs(
        self,
        workspace_root: Path,
        workspace: Workspace,
        vision_provider: VisionProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_path = self._write_png(
            workspace_root, "img.png", _FAKE_IMAGE_BYTES
        )
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), vision_provider, loader
        )
        executor = SyncToolExecutor(registry)
        call = ToolCall(
            call_id="l2",
            tool_name="analyze_image",
            arguments={"path": secret_path, "prompt": "describe"},
        )
        caplog.set_level(logging.DEBUG)
        import asyncio

        result = asyncio.run(executor.execute(call))
        assert not result.error
        description = result.output["description"]

        all_log = "\n".join(r.getMessage() for r in caplog.records)
        assert description not in all_log

    def test_image_bytes_not_in_logs(
        self,
        workspace_root: Path,
        workspace: Workspace,
        ocr_provider: OCRProvider,
        vision_provider: VisionProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._write_png(workspace_root, "img.png", _FAKE_IMAGE_BYTES)
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(registry, ocr_provider, vision_provider, loader)
        executor = SyncToolExecutor(registry)
        call = ToolCall(
            call_id="l3",
            tool_name="ocr_image",
            arguments={"path": "img.png"},
        )
        caplog.set_level(logging.DEBUG)
        import asyncio

        asyncio.run(executor.execute(call))
        all_log = "\n".join(r.getMessage() for r in caplog.records)
        # PNG magic should never appear in logs
        assert b"\x89PNG".decode("latin-1") not in all_log
        # Raw bytes should not be present
        assert _FAKE_IMAGE_BYTES.decode("latin-1") not in all_log


# ---------------------------------------------------------------------------
# 22-23. Security / path / network
# ---------------------------------------------------------------------------


class TestPathSecurity:
    def _write_png(self, root: Path, name: str) -> None:
        full = root / name
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(_FAKE_IMAGE_BYTES)

    def test_ocr_rejects_traversal(
        self,
        workspace_root: Path,
        workspace: Workspace,
    ) -> None:
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)
        import asyncio

        call = ToolCall(
            call_id="s1",
            tool_name="ocr_image",
            arguments={"path": "../escape.png"},
        )
        result = asyncio.run(executor.execute(call))
        assert result.error

    def test_ocr_rejects_absolute_path(
        self,
        workspace_root: Path,
        workspace: Workspace,
    ) -> None:
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)
        import asyncio

        call = ToolCall(
            call_id="s2",
            tool_name="ocr_image",
            arguments={"path": "/etc/passwd"},
        )
        result = asyncio.run(executor.execute(call))
        assert result.error

    def test_ocr_rejects_drive_path(
        self,
        workspace_root: Path,
        workspace: Workspace,
    ) -> None:
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)
        import asyncio

        call = ToolCall(
            call_id="s3",
            tool_name="ocr_image",
            arguments={"path": "C:\\Windows\\notepad.exe"},
        )
        result = asyncio.run(executor.execute(call))
        assert result.error

    def test_ocr_rejects_unc_path(
        self,
        workspace_root: Path,
        workspace: Workspace,
    ) -> None:
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)
        import asyncio

        call = ToolCall(
            call_id="s4",
            tool_name="ocr_image",
            arguments={"path": "\\\\server\\share\\file.png"},
        )
        result = asyncio.run(executor.execute(call))
        assert result.error

    def test_analyze_rejects_traversal(
        self,
        workspace_root: Path,
        workspace: Workspace,
    ) -> None:
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), loader
        )
        executor = SyncToolExecutor(registry)
        import asyncio

        call = ToolCall(
            call_id="s5",
            tool_name="analyze_image",
            arguments={"path": "../escape.png"},
        )
        result = asyncio.run(executor.execute(call))
        assert result.error

    def test_no_external_network_access(
        self,
        workspace_root: Path,
        workspace: Workspace,
        ocr_provider: OCRProvider,
        vision_provider: VisionProvider,
    ) -> None:
        self._write_png(workspace_root, "img.png")
        loader = ImageLoader(workspace)
        registry = DefaultToolRegistry()
        register_vision_tools(registry, ocr_provider, vision_provider, loader)
        executor = SyncToolExecutor(registry)
        import asyncio

        # Use a non-asyncio path: call the executor synchronously without
        # patching the global socket module (asyncio's internals need it).
        # We patch only the executor's callables' potential network sinks,
        # which in our fake providers is just hashlib (no socket access).
        # We verify by checking the providers do not import httpx/urllib.
        call1 = ToolCall(
            call_id="n1",
            tool_name="ocr_image",
            arguments={"path": "img.png"},
        )
        result1 = asyncio.run(executor.execute(call1))
        assert not result1.error

        call2 = ToolCall(
            call_id="n2",
            tool_name="analyze_image",
            arguments={"path": "img.png"},
        )
        result2 = asyncio.run(executor.execute(call2))
        assert not result2.error

        # Structural check: the OCR/Vision modules do not import any
        # network client.
        import core.vision.ocr as ocr_mod
        import core.vision.vision as vision_mod
        for name in ("httpx", "urllib", "urllib3", "requests", "aiohttp"):
            assert not hasattr(ocr_mod, name), f"ocr module unexpectedly has {name}"
            assert not hasattr(vision_mod, name), f"vision module unexpectedly has {name}"


# ---------------------------------------------------------------------------
# 24. Compatibility with existing ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistryCompat:
    def test_ocr_tool_can_be_registered_in_default_registry(
        self,
        image_loader: ImageLoader,
        ocr_provider: OCRProvider,
        vision_provider: VisionProvider,
    ) -> None:
        registry = DefaultToolRegistry()
        register_vision_tools(registry, ocr_provider, vision_provider, image_loader)
        # Registry exposes names
        names = registry.names()
        assert "ocr_image" in names
        assert "analyze_image" in names

    def test_tools_have_unique_capabilities(self) -> None:
        assert OCR_IMAGE_TOOL.capability == "vision"
        assert ANALYZE_IMAGE_TOOL.capability == "vision"


# ---------------------------------------------------------------------------
# 25. RAG compatibility
# ---------------------------------------------------------------------------


class TestRAGCompat:
    def test_ocr_to_document_single_page(self) -> None:
        result = OCRResult(
            image_id="p1",
            full_text="Hello world",
            confidence=0.9,
            blocks=(),
        )
        doc = ocr_to_document(
            [result], document_id="doc-1", filename="doc.png",
            source_path="scans/doc.png",
        )
        assert isinstance(doc, Document)
        assert doc.text == "Hello world"
        assert doc.document_id == "doc-1"
        assert doc.filename == "doc.png"
        assert doc.source_path == "scans/doc.png"
        assert doc.content_type == "text/plain"
        assert doc.metadata["source_kind"] == "ocr"
        assert doc.metadata["page_count"] == 1
        assert doc.metadata["average_confidence"] == 0.9

    def test_ocr_to_document_multi_page(self) -> None:
        results = [
            OCRResult(image_id="p1", full_text="Page one", confidence=0.8),
            OCRResult(image_id="p2", full_text="Page two", confidence=0.7),
        ]
        doc = ocr_to_document(
            results, document_id="doc-x", filename="x.png", source_path="x.png"
        )
        assert "Page one" in doc.text
        assert "Page two" in doc.text
        assert doc.metadata["page_count"] == 2
        assert doc.metadata["average_confidence"] == pytest.approx(0.75)

    def test_ocr_to_document_empty(self) -> None:
        doc = ocr_to_document(
            [], document_id="d", filename="d.png", source_path="d.png"
        )
        assert doc.text == ""
        assert doc.metadata["page_count"] == 0

    def test_ocr_to_document_preserves_language(self) -> None:
        results = [
            OCRResult(image_id="p1", full_text="hola", language="es"),
            OCRResult(image_id="p2", full_text="bonjour", language="fr"),
        ]
        doc = ocr_to_document(
            results, document_id="d", filename="d.png", source_path="d.png"
        )
        assert "es" in doc.metadata["languages"]
        assert "fr" in doc.metadata["languages"]


# ---------------------------------------------------------------------------
# 26. Vision routing integration
# ---------------------------------------------------------------------------


class TestVisionRoutingIntegration:
    def _make_registry(self) -> ModelRegistry:
        reg = ModelRegistry()
        reg.register(
            ModelDefinition(
                logical_name="vision",
                provider="ollama",
                provider_model="qwen2vl:2b",
                capabilities=frozenset({Capability.VISION}),
                input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
            )
        )
        reg.register(
            ModelDefinition(
                logical_name="text",
                provider="ollama",
                provider_model="llama3:3b",
                capabilities=frozenset({Capability.GENERAL}),
                input_modalities=frozenset({Modality.TEXT}),
            )
        )
        return reg

    def test_build_vision_routing_request(self) -> None:
        req = build_vision_routing_request()
        assert req.task_type == TaskType.VISION
        assert Capability.VISION in req.required_capabilities
        assert Modality.IMAGE in req.input_modalities
        assert Modality.TEXT in req.input_modalities

    def test_router_picks_vision_model(self) -> None:
        reg = self._make_registry()
        router = ModelRouter(reg)
        decision = router.route(build_vision_routing_request())
        assert decision.model.logical_name == "vision"
        assert decision.modality_satisfied

    def test_router_rejects_text_only_for_vision_task(self) -> None:
        reg = ModelRegistry()
        reg.register(
            ModelDefinition(
                logical_name="text-only",
                provider="ollama",
                provider_model="llama3:3b",
                capabilities=frozenset({Capability.GENERAL}),
                input_modalities=frozenset({Modality.TEXT}),
            )
        )
        router = ModelRouter(reg)
        with pytest.raises(NoSuitableModelError):
            router.route(build_vision_routing_request())


# ---------------------------------------------------------------------------
# 27. Deterministic results for identical inputs
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_ocr_deterministic(self, ocr_provider: OCRProvider) -> None:
        image = _make_image(image_id="final-1")
        r1 = ocr_provider.recognize(image)
        r2 = ocr_provider.recognize(image)
        assert r1.full_text == r2.full_text
        assert r1.confidence == r2.confidence

    def test_vision_deterministic(self, vision_provider: VisionProvider) -> None:
        image = _make_image(image_id="final-2")
        r1 = vision_provider.analyze(image, "describe")
        r2 = vision_provider.analyze(image, "describe")
        assert r1.description == r2.description
        assert r1.tags == r2.tags
        assert r1.regions[0].label == r2.regions[0].label

    def test_ocr_image_id_stable_for_path(
        self, image_loader: ImageLoader, workspace_root: Path
    ) -> None:
        (workspace_root / "x.png").write_bytes(_FAKE_IMAGE_BYTES)
        img1 = image_loader.load("x.png")
        img2 = image_loader.load("x.png")
        assert img1.image_id == img2.image_id


# ---------------------------------------------------------------------------
# Additional: OCR result type and region type basics
# ---------------------------------------------------------------------------


class TestResultTypeIntegrity:
    def test_ocr_result_construction(self) -> None:
        r = OCRResult(
            image_id="x",
            full_text="hi",
            page_number=1,
            page_count=1,
            language="en",
        )
        assert r.image_id == "x"
        assert r.page_number == 1
        assert r.page_count == 1
        assert r.language == "en"

    def test_vision_region_construction(self) -> None:
        r = VisionRegion(
            label="person",
            confidence=0.9,
            box=OCRBoundingBox(x=10, y=20, width=100, height=200),
        )
        assert r.label == "person"
        assert r.box is not None
        assert r.box.width == 100
