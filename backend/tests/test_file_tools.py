"""Comprehensive tests for Phase 5A file tools.

All tests run entirely offline against temporary directories. They never
touch the real user's filesystem.

The test file is organised into the following sections:

* :class:`TestWorkspace` — workspace construction and path validation.
* :class:`TestListFiles` — the ``list_files`` tool.
* :class:`TestReadFile` — the ``read_file`` tool.
* :class:`TestWriteFile` — the ``write_file`` tool.
* :class:`TestToolAbstraction` — registering tools with the agent's
  ToolRegistry.
* :class:`TestAgentIntegration` — agent + tool registry + file tool
  end-to-end with a fake router and planner.
* :class:`TestSensitiveLogging` — file contents are NOT logged.
* :class:`TestConcurrentReads` — concurrent reads of the same file.
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock

import pytest

from core.agent import (
    Agent,
    AgentConfig,
    AgentStatus,
    ExecutionPhase,
    Plan,
    Planner,
    PlanStep,
    SimpleVerifier,
    SyncToolExecutor,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    Verifier,
)
from core.agent.errors import UnknownToolError
from core.agent.registry import DefaultToolRegistry
from core.agent.types import AgentState, VerificationResult
from core.tools import (
    FileNotFoundError,
    FileToolError,
    LIST_FILES_TOOL,
    PermissionDeniedError,
    READ_FILE_TOOL,
    UnsupportedFileError,
    WRITE_FILE_TOOL,
    Workspace,
    WRITE_FILE_TOOL as _WRITE_FILE_TOOL_DUPLICATE,  # noqa: F401 — exported check
)
from core.tools.file_tools import FileToolExecutor
from core.tools.registry import register_file_tools
from core.tools.types import FileListItem
from core.tools.workspace import (
    AbsolutePathError,
    InvalidPathError,
    PathTraversalError,
    SymlinkEscapeError,
    WorkspaceError,
)
from core.routing.types import (
    Capability,
    ModelDefinition,
    RoutingDecision,
    RoutingRequest,
    TaskType,
)


# Sentinel import: confirm duplicate naming doesn't shadow
assert _WRITE_FILE_TOOL_DUPLICATE is WRITE_FILE_TOOL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_root() -> Iterator[Path]:
    """A temporary workspace root, removed after the test.

    Tests may freely create files and directories inside this path.
    """
    with tempfile.TemporaryDirectory(prefix="sovereign_ai_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def workspace(workspace_root: Path) -> Workspace:
    """A :class:`Workspace` rooted at the temporary directory."""
    return Workspace(root_path=str(workspace_root))


@pytest.fixture
def file_executor(workspace: Workspace) -> FileToolExecutor:
    """A :class:`FileToolExecutor` bound to the temporary workspace."""
    return FileToolExecutor(workspace)


@pytest.fixture
def file_registry(workspace: Workspace) -> DefaultToolRegistry:
    """A :class:`DefaultToolRegistry` with all three file tools registered."""
    registry = DefaultToolRegistry()
    register_file_tools(registry, workspace)
    return registry


def make_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    call_id: str = "test-call-0",
) -> ToolCall:
    """Construct a :class:`ToolCall` for testing."""
    return ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments or {},
    )


# ---------------------------------------------------------------------------
# Workspace security
# ---------------------------------------------------------------------------


class TestWorkspace:
    def test_constructs_with_existing_directory(self, workspace_root: Path) -> None:
        ws = Workspace(root_path=str(workspace_root))
        assert ws.name == workspace_root.name

    def test_rejects_nonexistent_root(self, workspace_root: Path) -> None:
        with pytest.raises(WorkspaceError):
            Workspace(root_path=str(workspace_root / "does_not_exist"))

    def test_rejects_file_as_root(self, workspace_root: Path) -> None:
        f = workspace_root / "a_file.txt"
        f.write_text("not a directory")
        with pytest.raises(WorkspaceError):
            Workspace(root_path=str(f))

    def test_rejects_tilde_root(self, workspace_root: Path) -> None:
        with pytest.raises(WorkspaceError):
            Workspace(root_path="~/somewhere")

    def test_rejects_empty_root(self) -> None:
        with pytest.raises(WorkspaceError):
            Workspace(root_path="")

    def test_resolves_legitimate_path(self, workspace: Workspace, workspace_root: Path) -> None:
        target = workspace_root / "file.txt"
        target.write_text("hello")
        resolved = workspace.resolve("file.txt")
        assert resolved == target.resolve()

    def test_resolves_nested_path(self, workspace: Workspace, workspace_root: Path) -> None:
        nested = workspace_root / "a" / "b" / "c.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("nested")
        resolved = workspace.resolve("a/b/c.txt")
        assert resolved == nested.resolve()

    def test_rejects_parent_traversal(self, workspace: Workspace) -> None:
        with pytest.raises(PathTraversalError):
            workspace.resolve("../etc/passwd")

    def test_rejects_parent_traversal_windows_style(self, workspace: Workspace) -> None:
        with pytest.raises(PathTraversalError):
            workspace.resolve("..\\Windows\\notepad.exe")

    def test_rejects_double_parent_traversal(self, workspace: Workspace) -> None:
        with pytest.raises(PathTraversalError):
            workspace.resolve("a/../../b")

    def test_rejects_absolute_posix_path(self, workspace: Workspace) -> None:
        with pytest.raises(AbsolutePathError):
            workspace.resolve("/etc/passwd")

    def test_rejects_windows_drive_path(self, workspace: Workspace) -> None:
        with pytest.raises(AbsolutePathError):
            workspace.resolve("C:\\Windows\\notepad.exe")
        with pytest.raises(AbsolutePathError):
            workspace.resolve("D:/data")

    def test_rejects_unc_path(self, workspace: Workspace) -> None:
        with pytest.raises(AbsolutePathError):
            workspace.resolve("\\\\server\\share\\file")

    def test_rejects_empty_path(self, workspace: Workspace) -> None:
        with pytest.raises(InvalidPathError):
            workspace.resolve("")

    def test_rejects_whitespace_only_path(self, workspace: Workspace) -> None:
        with pytest.raises(InvalidPathError):
            workspace.resolve("   ")

    def test_rejects_path_with_nul(self, workspace: Workspace) -> None:
        with pytest.raises(InvalidPathError):
            workspace.resolve("file\x00.txt")

    def test_rejects_extremely_long_path(self, workspace: Workspace) -> None:
        long_path = "a/" * 5000 + "file.txt"
        with pytest.raises(InvalidPathError):
            workspace.resolve(long_path)

    def test_rejects_control_characters(self, workspace: Workspace) -> None:
        with pytest.raises(InvalidPathError):
            workspace.resolve("file\x01.txt")

    def test_relative_to_returns_forward_slash_path(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        target = workspace_root / "sub" / "file.txt"
        target.parent.mkdir()
        target.write_text("hi")
        rel = workspace.relative_to(target.resolve())
        assert rel == "sub/file.txt"
        assert "\\" not in rel

    def test_relative_to_root(self, workspace: Workspace) -> None:
        rel = workspace.relative_to(workspace._canonical_root)  # noqa: SLF001
        assert rel == ""

    def test_relative_to_outside_raises(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        # Use a sibling dir of the workspace as a path that is *guaranteed*
        # to be outside the workspace.
        with pytest.raises(WorkspaceError):
            workspace.relative_to(str(workspace_root.parent / "definitely_outside.txt"))


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


class TestListFiles:
    @pytest.mark.asyncio
    async def test_list_empty_workspace(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(make_call("list_files"))
        assert not result.error
        assert result.output == []

    @pytest.mark.asyncio
    async def test_list_files_and_dirs(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        (workspace_root / "a.txt").write_text("a")
        (workspace_root / "b.txt").write_text("b")
        (workspace_root / "subdir").mkdir()
        result = await file_executor.execute(make_call("list_files"))
        assert not result.error
        names = {item["name"] for item in result.output}
        assert names == {"a.txt", "b.txt", "subdir"}

    @pytest.mark.asyncio
    async def test_list_subdirectory(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        sub = workspace_root / "sub"
        sub.mkdir()
        (sub / "x.txt").write_text("x")
        result = await file_executor.execute(
            make_call("list_files", {"directory": "sub"})
        )
        assert not result.error
        names = {item["name"] for item in result.output}
        assert names == {"x.txt"}

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("list_files", {"directory": "nope"})
        )
        assert result.error
        assert "not found" in result.error_message.lower() or "FileNotFoundError" in result.error_message

    @pytest.mark.asyncio
    async def test_list_rejects_parent_traversal(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("list_files", {"directory": ".."})
        )
        assert result.error
        assert "not allowed" in result.error_message.lower() or "traversal" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_list_rejects_absolute_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("list_files", {"directory": "/etc"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_list_with_path_that_is_file(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "afile.txt"
        f.write_text("x")
        result = await file_executor.execute(
            make_call("list_files", {"directory": "afile.txt"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_list_includes_safe_metadata(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        (workspace_root / "x.txt").write_text("hello")
        result = await file_executor.execute(make_call("list_files"))
        assert not result.error
        item = result.output[0]
        assert item["name"] == "x.txt"
        assert item["path"] == "x.txt"
        assert item["is_file"] is True
        assert item["size_bytes"] == 5
        assert item["is_text"] is True

    @pytest.mark.asyncio
    async def test_list_output_paths_are_relative_only(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        """The list_files output must not expose absolute host paths."""
        (workspace_root / "x.txt").write_text("x")
        result = await file_executor.execute(make_call("list_files"))
        for item in result.output:
            path = item["path"]
            assert not os.path.isabs(path)
            assert not path.startswith("\\\\")
            # The path should not contain the absolute workspace root
            assert str(workspace_root.resolve()) not in path


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_text_file(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "doc.txt"
        f.write_text("hello world")
        result = await file_executor.execute(
            make_call("read_file", {"path": "doc.txt"})
        )
        assert not result.error
        assert result.output["content"] == "hello world"
        assert result.output["size_bytes"] == 11
        assert result.output["encoding"] == "utf-8"
        assert result.output["path"] == "doc.txt"

    @pytest.mark.asyncio
    async def test_read_empty_file(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "empty.txt"
        f.write_text("")
        result = await file_executor.execute(
            make_call("read_file", {"path": "empty.txt"})
        )
        assert not result.error
        assert result.output["content"] == ""
        assert result.output["size_bytes"] == 0

    @pytest.mark.asyncio
    async def test_read_unicode_content(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "uni.txt"
        f.write_text("héllo wörld 🌍", encoding="utf-8")
        result = await file_executor.execute(
            make_call("read_file", {"path": "uni.txt"})
        )
        assert not result.error
        assert "🌍" in result.output["content"]

    @pytest.mark.asyncio
    async def test_read_unicode_filename(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "文档.txt"
        f.write_text("content")
        result = await file_executor.execute(
            make_call("read_file", {"path": "文档.txt"})
        )
        assert not result.error
        assert result.output["content"] == "content"

    @pytest.mark.asyncio
    async def test_read_filename_with_spaces(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "name with spaces.txt"
        f.write_text("ok")
        result = await file_executor.execute(
            make_call("read_file", {"path": "name with spaces.txt"})
        )
        assert not result.error
        assert result.output["content"] == "ok"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("read_file", {"path": "missing.txt"})
        )
        assert result.error
        assert "not found" in result.error_message.lower() or "FileNotFoundError" in result.error_message

    @pytest.mark.asyncio
    async def test_read_rejects_directory(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        (workspace_root / "subdir").mkdir()
        result = await file_executor.execute(
            make_call("read_file", {"path": "subdir"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_binary_extension(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = await file_executor.execute(
            make_call("read_file", {"path": "image.png"})
        )
        assert result.error
        assert "UnsupportedFileError" in result.error_message

    @pytest.mark.asyncio
    async def test_read_rejects_pdf(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "doc.pdf"
        f.write_bytes(b"%PDF-1.4\n%binary\n")
        result = await file_executor.execute(
            make_call("read_file", {"path": "doc.pdf"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_zip(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "x.zip"
        f.write_bytes(b"PK\x03\x04binary")
        result = await file_executor.execute(
            make_call("read_file", {"path": "x.zip"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_unsupported_extension(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "x.unknownext"
        f.write_text("hi")
        result = await file_executor.execute(
            make_call("read_file", {"path": "x.unknownext"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_binary_content_with_text_extension(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "fake.txt"
        f.write_bytes(b"\x00\x01\x02\x03binary data")
        result = await file_executor.execute(
            make_call("read_file", {"path": "fake.txt"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_parent_traversal(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("read_file", {"path": "../etc/passwd"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_absolute_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("read_file", {"path": "/etc/passwd"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_drive_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("read_file", {"path": "C:\\Windows\\notepad.exe"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_rejects_unc_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("read_file", {"path": "\\\\server\\share\\file.txt"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_read_requires_path_argument(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(make_call("read_file", {}))
        assert result.error

    @pytest.mark.asyncio
    async def test_read_nested_file(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        nested = workspace_root / "a" / "b" / "c" / "deep.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("deep content")
        result = await file_executor.execute(
            make_call("read_file", {"path": "a/b/c/deep.txt"})
        )
        assert not result.error
        assert result.output["content"] == "deep content"

    @pytest.mark.asyncio
    async def test_read_large_but_reasonable_file(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        # 1 MiB file — well within the 10 MiB read limit
        content = "x" * (1024 * 1024)
        f = workspace_root / "large.txt"
        f.write_text(content)
        result = await file_executor.execute(
            make_call("read_file", {"path": "large.txt"})
        )
        assert not result.error
        assert len(result.output["content"]) == 1024 * 1024

    @pytest.mark.asyncio
    async def test_read_oversized_file_rejected(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        # 12 MiB file — exceeds the 10 MiB read limit
        # We don't actually write 12 MiB to disk in the test, we just
        # write a small placeholder and then resize it.
        f = workspace_root / "huge.txt"
        f.write_text("placeholder")
        # Use truncate to extend the file with NUL bytes (sparse-ish)
        with open(f, "ab") as fh:
            fh.truncate(12 * 1024 * 1024)
        result = await file_executor.execute(
            make_call("read_file", {"path": "huge.txt"})
        )
        assert result.error


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_simple_file(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        result = await file_executor.execute(
            make_call("write_file", {"path": "out.txt", "content": "hello"})
        )
        assert not result.error
        assert (workspace_root / "out.txt").read_text() == "hello"
        assert result.output["size_bytes"] == 5
        assert result.output["path"] == "out.txt"

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        f = workspace_root / "out.txt"
        f.write_text("original")
        result = await file_executor.execute(
            make_call("write_file", {"path": "out.txt", "content": "new"})
        )
        assert not result.error
        assert f.read_text() == "new"

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "a/b/c/out.txt", "content": "deep"},
            )
        )
        assert not result.error
        assert (workspace_root / "a" / "b" / "c" / "out.txt").read_text() == "deep"
        assert (workspace_root / "a").is_dir()
        assert (workspace_root / "a" / "b").is_dir()
        assert (workspace_root / "a" / "b" / "c").is_dir()
        # created_directories should list the new directories
        created = result.output["created_directories"]
        assert any("a" in d for d in created)

    @pytest.mark.asyncio
    async def test_write_unicode_content(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        content = "héllo wörld 🌍"
        result = await file_executor.execute(
            make_call("write_file", {"path": "u.txt", "content": content})
        )
        assert not result.error
        assert (workspace_root / "u.txt").read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_write_unicode_filename(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        result = await file_executor.execute(
            make_call("write_file", {"path": "文档.txt", "content": "ok"})
        )
        assert not result.error
        assert (workspace_root / "文档.txt").read_text() == "ok"

    @pytest.mark.asyncio
    async def test_write_filename_with_spaces(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "name with spaces.txt", "content": "ok"},
            )
        )
        assert not result.error
        assert (workspace_root / "name with spaces.txt").read_text() == "ok"

    @pytest.mark.asyncio
    async def test_write_rejects_parent_traversal(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "../escape.txt", "content": "bad"},
            )
        )
        assert result.error
        # The file should not have been created
        assert not (file_executor._workspace._canonical_root.parent / "escape.txt").exists()  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_write_rejects_absolute_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "/tmp/escape.txt", "content": "bad"},
            )
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_write_rejects_drive_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "C:\\Windows\\bad.txt", "content": "bad"},
            )
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_write_rejects_unc_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "\\\\server\\share\\bad.txt", "content": "bad"},
            )
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_write_oversized_content_rejected(
        self, file_executor: FileToolExecutor
    ) -> None:
        big_content = "x" * (11 * 1024 * 1024)  # 11 MiB > 10 MiB limit
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "huge.txt", "content": big_content},
            )
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_write_requires_content(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("write_file", {"path": "x.txt"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_write_requires_path(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call("write_file", {"content": "data"})
        )
        assert result.error

    @pytest.mark.asyncio
    async def test_write_empty_content(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": "empty.txt", "content": ""},
            )
        )
        assert not result.error
        assert (workspace_root / "empty.txt").exists()
        assert (workspace_root / "empty.txt").read_text() == ""

    @pytest.mark.asyncio
    async def test_write_rejects_when_create_dirs_disabled(
        self, workspace_root: Path
    ) -> None:
        ws = Workspace(root_path=str(workspace_root), allow_create_dirs=False)
        executor = FileToolExecutor(ws)
        result = await executor.execute(
            make_call(
                "write_file",
                {"path": "newdir/file.txt", "content": "data"},
            )
        )
        assert result.error
        # Verify nothing was created
        assert not (workspace_root / "newdir").exists()

    @pytest.mark.asyncio
    async def test_write_create_dirs_explicit_true(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {
                    "path": "explicit/sub/file.txt",
                    "content": "data",
                    "create_directories": True,
                },
            )
        )
        assert not result.error
        assert (workspace_root / "explicit" / "sub" / "file.txt").read_text() == "data"

    @pytest.mark.asyncio
    async def test_write_create_dirs_explicit_false(
        self, file_executor: FileToolExecutor
    ) -> None:
        result = await file_executor.execute(
            make_call(
                "write_file",
                {
                    "path": "no_creation/file.txt",
                    "content": "data",
                    "create_directories": False,
                },
            )
        )
        assert result.error


# ---------------------------------------------------------------------------
# Tool registry integration
# ---------------------------------------------------------------------------


class TestToolAbstraction:
    def test_list_files_tool_definition(self) -> None:
        assert LIST_FILES_TOOL.name == "list_files"
        assert LIST_FILES_TOOL.capability == "file_system"

    def test_read_file_tool_definition(self) -> None:
        assert READ_FILE_TOOL.name == "read_file"
        assert "path" in READ_FILE_TOOL.input_schema["required"]

    def test_write_file_tool_definition(self) -> None:
        assert WRITE_FILE_TOOL.name == "write_file"
        required = WRITE_FILE_TOOL.input_schema["required"]
        assert "path" in required
        assert "content" in required

    def test_register_file_tools_registers_all_three(
        self, workspace: Workspace
    ) -> None:
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        assert registry.has("list_files")
        assert registry.has("read_file")
        assert registry.has("write_file")

    def test_register_file_tools_via_sync_executor(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        (workspace_root / "x.txt").write_text("ok")
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        executor = SyncToolExecutor(registry)
        # Run a read_file through the SyncToolExecutor
        call = ToolCall(
            call_id="c1",
            tool_name="read_file",
            arguments={"path": "x.txt"},
        )
        # Use asyncio to run the async executor
        import asyncio
        result = asyncio.run(executor.execute(call))
        assert not result.error
        assert result.output["content"] == "ok"

    def test_register_file_tools_rejects_non_default_registry(
        self, workspace: Workspace
    ) -> None:
        class CustomRegistry(ToolRegistry):
            def register(self, tool, callable=None):  # type: ignore[override]
                pass

            def unregister(self, name): pass
            def has(self, name): return False
            def get(self, name): raise UnknownToolError(name)
            def names(self): return []

        with pytest.raises(TypeError):
            register_file_tools(CustomRegistry(), workspace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Workspace boundary enforcement
# ---------------------------------------------------------------------------


class TestWorkspaceBoundary:
    @pytest.mark.asyncio
    async def test_symlink_escape_rejected(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        """A symlink pointing outside the workspace must be rejected."""
        outside = workspace_root.parent / "sovereign_ai_test_outside.txt"
        outside.write_text("secret")
        link = workspace_root / "link"
        try:
            link.symlink_to(outside)
        except OSError:
            outside.unlink(missing_ok=True)
            pytest.skip("cannot create symlink here")
        try:
            result_workspace = Workspace(root_path=str(workspace_root))
            with pytest.raises(SymlinkEscapeError):
                result_workspace.resolve("link")
        finally:
            # Clean up
            link.unlink(missing_ok=True)
            outside.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_safe_symlink_allowed(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        """A symlink pointing inside the workspace must work."""
        if not _supports_symlinks():
            pytest.skip("symlinks not supported on this platform/filesystem")
        target = workspace_root / "target.txt"
        target.write_text("safe content")
        link = workspace_root / "link"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("cannot create symlink here")
        result = await file_executor.execute(
            make_call("read_file", {"path": "link"})
        )
        assert not result.error
        assert result.output["content"] == "safe content"

    @pytest.mark.asyncio
    async def test_cannot_resolve_outside_workspace_using_parent_segments(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        sibling = workspace_root.parent / "sibling_workspace"
        sibling.mkdir(exist_ok=True)
        target = sibling / "secret.txt"
        target.write_text("top secret")
        with pytest.raises(PathTraversalError):
            workspace.resolve(f"../sibling_workspace/secret.txt")
        # The file should still exist (we did not delete it)
        assert target.exists()

    @pytest.mark.asyncio
    async def test_write_cannot_create_files_outside_workspace(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        # Try to write a file with a traversal path
        result = await file_executor.execute(
            make_call(
                "write_file",
                {"path": f"../escape_{os.getpid()}.txt", "content": "bad"},
            )
        )
        assert result.error
        # Verify no file was created in the workspace's parent dir
        escape_path = workspace_root.parent / f"escape_{os.getpid()}.txt"
        assert not escape_path.exists()

    @pytest.mark.asyncio
    async def test_list_does_not_leak_absolute_paths(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        (workspace_root / "leak.txt").write_text("x")
        result = await file_executor.execute(make_call("list_files"))
        assert not result.error
        # Serialize the entire result and ensure no absolute path appears
        serialized = str(result.output)
        assert str(workspace_root.resolve()) not in serialized
        assert "C:\\" not in serialized
        assert "\\\\" not in serialized


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


class _FakeRouter:
    """A minimal fake router that returns a configured model."""

    def __init__(self, model_name: str = "general") -> None:
        self.model_name = model_name
        self.calls: list[RoutingRequest] = []

    def route(self, request: RoutingRequest) -> RoutingDecision:
        self.calls.append(request)
        model = ModelDefinition(
            logical_name=self.model_name,
            provider="fake",
            provider_model="fake-model",
            capabilities=frozenset({Capability.GENERAL}),
        )
        return RoutingDecision(
            model=model,
            reason=f"fake router selected {self.model_name}",
            score=100,
            matched_capabilities=frozenset({Capability.GENERAL}),
            modality_satisfied=True,
            preferred_honoured=False,
        )


class _ScriptedPlanner(Planner):
    """A planner that returns a fixed list of plan steps."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.calls: list[Any] = []

    async def plan(
        self,
        task: str,
        available_tools: list[str],
        state: AgentState,
    ) -> Plan:
        self.calls.append((task, list(available_tools)))
        return self._plan


class _PassVerifier(Verifier):
    async def verify(self, state: AgentState) -> VerificationResult:
        return VerificationResult(passed=True, reason="test ok")


class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_agent_uses_file_tool_via_registry(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        # Set up: a file to read
        (workspace_root / "hello.txt").write_text("integration ok")

        # Register tools
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        executor = SyncToolExecutor(registry)

        # Plan: a single read_file step
        plan = Plan(
            goal="read hello.txt",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="read hello.txt",
                    tool_name="read_file",
                    inputs={"path": "hello.txt"},
                ),
            ),
        )

        agent = Agent(
            model_router=_FakeRouter(),
            tool_executor=executor,
            planner=_ScriptedPlanner(plan),
            verifier=_PassVerifier(),
            config=AgentConfig(max_iterations=5, max_tool_calls=10),
            tool_registry=registry,
        )

        result = await agent.run("read hello.txt")
        assert result.status == AgentStatus.COMPLETE
        assert len(result.observations) == 1
        obs = result.observations[0]
        assert "integration ok" in obs.content

    @pytest.mark.asyncio
    async def test_agent_writes_file_via_registry(
        self, workspace: Workspace, workspace_root: Path
    ) -> None:
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        executor = SyncToolExecutor(registry)

        plan = Plan(
            goal="write a file",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="write out.txt",
                    tool_name="write_file",
                    inputs={"path": "out.txt", "content": "agent wrote this"},
                ),
            ),
        )

        agent = Agent(
            model_router=_FakeRouter(),
            tool_executor=executor,
            planner=_ScriptedPlanner(plan),
            verifier=_PassVerifier(),
            config=AgentConfig(max_iterations=5, max_tool_calls=10),
            tool_registry=registry,
        )

        result = await agent.run("write a file")
        assert result.status == AgentStatus.COMPLETE
        assert (workspace_root / "out.txt").read_text() == "agent wrote this"

    @pytest.mark.asyncio
    async def test_agent_tool_failure_continues(
        self, workspace: Workspace
    ) -> None:
        """A tool failure should be recorded as an observation, not crash the agent."""
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        executor = SyncToolExecutor(registry)

        plan = Plan(
            goal="read missing file",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="read missing",
                    tool_name="read_file",
                    inputs={"path": "no_such_file.txt"},
                ),
            ),
        )

        agent = Agent(
            model_router=_FakeRouter(),
            tool_executor=executor,
            planner=_ScriptedPlanner(plan),
            verifier=_PassVerifier(),
            config=AgentConfig(max_iterations=5, max_tool_calls=10),
            tool_registry=registry,
        )

        result = await agent.run("read missing")
        # The agent should still complete (the verifier passes) but record
        # the tool failure as an observation.
        assert result.status == AgentStatus.COMPLETE
        assert any(
            "not found" in obs.content.lower() or "FileNotFound" in obs.content
            for obs in result.observations
        )

    @pytest.mark.asyncio
    async def test_agent_security_violation_caught(
        self, workspace: Workspace
    ) -> None:
        """A path-traversal attempt from a plan must be rejected by the tool."""
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        executor = SyncToolExecutor(registry)

        plan = Plan(
            goal="read escape",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="read escape",
                    tool_name="read_file",
                    inputs={"path": "../etc/passwd"},
                ),
            ),
        )

        agent = Agent(
            model_router=_FakeRouter(),
            tool_executor=executor,
            planner=_ScriptedPlanner(plan),
            verifier=_PassVerifier(),
            config=AgentConfig(max_iterations=5, max_tool_calls=10),
            tool_registry=registry,
        )

        result = await agent.run("read escape")
        assert result.status == AgentStatus.COMPLETE
        # The failure should be in the observations
        assert any("not allowed" in obs.content.lower() for obs in result.observations)


# ---------------------------------------------------------------------------
# Sensitive logging
# ---------------------------------------------------------------------------


class TestSensitiveLogging:
    @pytest.mark.asyncio
    async def test_file_contents_not_logged(
        self,
        file_executor: FileToolExecutor,
        workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "TOP-SECRET-PASSWORD-9876"
        (workspace_root / "secret.txt").write_text(secret)
        caplog.set_level(logging.DEBUG)
        await file_executor.execute(make_call("read_file", {"path": "secret.txt"}))
        for record in caplog.records:
            assert secret not in record.getMessage()

    @pytest.mark.asyncio
    async def test_failure_messages_do_not_leak_content(
        self,
        file_executor: FileToolExecutor,
        workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Even on failure, file contents must not appear in logs
        secret_content = "API-KEY-WITH-SECRET-DATA-5555"
        f = workspace_root / "leak.txt"
        # Use a binary file disguised as text
        f.write_bytes(secret_content.encode() + b"\x00\x01\x02")
        caplog.set_level(logging.DEBUG)
        await file_executor.execute(make_call("read_file", {"path": "leak.txt"}))
        for record in caplog.records:
            assert secret_content not in record.getMessage()

    @pytest.mark.asyncio
    async def test_absolute_paths_not_logged(
        self,
        file_executor: FileToolExecutor,
        workspace_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (workspace_root / "x.txt").write_text("data")
        caplog.set_level(logging.DEBUG)
        await file_executor.execute(make_call("read_file", {"path": "x.txt"}))
        workspace_abs = str(workspace_root.resolve())
        for record in caplog.records:
            # The absolute workspace path should not appear in logs
            assert workspace_abs not in record.getMessage()


# ---------------------------------------------------------------------------
# Concurrent reads
# ---------------------------------------------------------------------------


class TestConcurrentReads:
    @pytest.mark.asyncio
    async def test_concurrent_reads_of_same_file(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        """Reading the same file from many coroutines should be safe."""
        f = workspace_root / "concurrent.txt"
        # Use binary write to avoid Windows newline translation so that
        # the read result exactly matches the input.
        content_bytes = b"concurrent content\n" * 100
        f.write_bytes(content_bytes)
        expected = content_bytes.decode("utf-8")
        calls = [
            file_executor.execute(
                make_call("read_file", {"path": "concurrent.txt"}, call_id=f"c{i}")
            )
            for i in range(20)
        ]
        results = await asyncio.gather(*calls)
        for r in results:
            assert not r.error
            assert r.output["content"] == expected

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_different_files(
        self, file_executor: FileToolExecutor, workspace_root: Path
    ) -> None:
        """Writing to different files concurrently should be safe."""
        calls = [
            file_executor.execute(
                make_call(
                    "write_file",
                    {"path": f"file_{i}.txt", "content": f"content {i}"},
                    call_id=f"c{i}",
                )
            )
            for i in range(20)
        ]
        results = await asyncio.gather(*calls)
        for i, r in enumerate(results):
            assert not r.error
            assert (workspace_root / f"file_{i}.txt").read_text() == f"content {i}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _supports_symlinks() -> bool:
    """Return True if the current filesystem supports symlinks.

    On Windows, symlink creation typically requires admin privileges or
    developer mode. We skip symlink tests when creation fails.
    """
    if sys.platform == "win32":
        # Windows: we try to create one and let the test skip on failure
        return True
    return True
