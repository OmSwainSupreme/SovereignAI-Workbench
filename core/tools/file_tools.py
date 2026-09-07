"""File tools (Phase 5A) — list_files, read_file, write_file.

These tools operate ONLY inside an explicitly configured :class:`Workspace`.
All path validation is delegated to the workspace; the tools themselves do
not perform any path canonicalisation or traversal checks.

Security summary
----------------

* Every tool argument is passed through :meth:`Workspace.resolve` first.
* The executor never touches paths outside the workspace.
* Reads of unsupported files (binary, wrong extension) are rejected
  *before* attempting to decode them.
* Writes may create intermediate directories only if the workspace was
  constructed with ``allow_create_dirs=True``.
* All errors are returned as :class:`ToolResult` with ``error=True`` and
  a non-sensitive ``error_message``; the tool never raises to the
  caller.

Tool contracts
--------------

list_files(directory)
    List immediate children of ``directory`` (default: workspace root).
    Returns a list of :class:`FileListItem` with workspace-relative
    paths. Subdirectories are not recursed.

read_file(path)
    Read a text file. Returns :class:`ReadResult` with the content.
    Binary files, oversized files, and unsupported extensions are
    rejected with :class:`UnsupportedFileError`.

write_file(path, content, create_directories=None)
    Write text content to a file. ``content`` must be a string.
    Intermediate directories are created only if explicitly allowed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from core.agent.errors import ToolExecutionError
from core.agent.interfaces import ToolDefinition, ToolExecutor
from core.agent.registry import DefaultToolRegistry
from core.agent.types import ToolCall, ToolResult
from core.tools.types import (
    FileListItem,
    FileNotFoundError,
    FileToolError,
    PermissionDeniedError,
    ReadResult,
    UnsupportedFileError,
    WriteResult,
)
from core.tools.workspace import (
    AbsolutePathError,
    InvalidPathError,
    PathTraversalError,
    SymlinkEscapeError,
    Workspace,
    WorkspaceError,
)


_logger = logging.getLogger("sovereign-ai.tools.files")

# Maximum size (in bytes) of a file that read_file will read. This is a
# conservative guard against accidentally reading huge files. Real
# production systems may want to make this configurable per workspace.
_MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MiB

# Maximum size (in bytes) of a single write_file invocation. Same
# rationale as above.
_MAX_WRITE_BYTES = 10 * 1024 * 1024  # 10 MiB

# Maximum number of entries returned by a single list_files call.
# Anything larger is a sign of a misconfigured workspace.
_MAX_LIST_ENTRIES = 10_000

# Encoding used for text I/O. UTF-8 is the default; errors are strict so
# a binary file masquerading as text surfaces as a clean error.
_TEXT_ENCODING = "utf-8"

# Extensions that we DO support. Conservative: any extension outside
# this set is rejected by read_file BEFORE we attempt to open the
# file. This avoids accidentally decoding binary data as text.
_SUPPORTED_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".py",  # Python source is text
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
        ".tsv",
        ".log",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".js",
        ".ts",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".gitignore",
        ".env",
    }
)

# Extensions that are clearly binary. Used to give a faster, clearer
# error than the "open and try to decode" fallback.
_KNOWN_BINARY_EXTENSIONS = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff",
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".exe", ".dll", ".so", ".dylib",
        ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
        ".bin", ".iso", ".img",
    }
)

# Maximum number of bytes to read from the start of a file when
# sniffing for binary content. If any of these bytes are NUL or a
# high-bit byte that is not valid UTF-8, we treat the file as binary.
_BINARY_SNIFF_BYTES = 8192


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


LIST_FILES_TOOL = ToolDefinition(
    name="list_files",
    description=(
        "List immediate files and directories inside a workspace directory. "
        "Returns workspace-relative metadata. Does not recurse."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": (
                    "Workspace-relative directory to list. "
                    "Default: workspace root. Must be inside the workspace."
                ),
            },
        },
    },
    output_description=(
        "A list of {name, path, is_file, size_bytes, mime_hint, is_text}."
    ),
    capability="file_system",
)


READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description=(
        "Read the text content of a file inside the workspace. "
        "Supports common text formats only; binary files are rejected."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to the file.",
            },
        },
        "required": ["path"],
    },
    output_description=(
        "A {path, content, size_bytes, encoding} object."
    ),
    capability="file_system",
)


WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description=(
        "Write text content to a file inside the workspace. "
        "Overwrites existing files. May create intermediate directories "
        "if the workspace is configured to allow it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write. Must be a string.",
            },
            "create_directories": {
                "type": "boolean",
                "description": (
                    "Whether to create intermediate directories if they "
                    "do not exist. Defaults to the workspace's policy."
                ),
            },
        },
        "required": ["path", "content"],
    },
    output_description=(
        "A {path, size_bytes, created_directories} object."
    ),
    capability="file_system",
)


# ---------------------------------------------------------------------------
# FileToolExecutor
# ---------------------------------------------------------------------------


class FileToolExecutor(ToolExecutor):
    """A :class:`ToolExecutor` that runs the three Phase 5A file tools.

    The executor is bound to a single :class:`Workspace`. The workspace
    is the source of truth for path validation; this class only translates
    tool arguments into workspace calls and packages the results.

    The executor is safe for concurrent use at the instance level (it
    holds only a reference to the immutable workspace), but individual
    file operations are not atomic. Callers that need atomic
    read-modify-write semantics must serialise at a higher level.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    # ------------------------------------------------------------------ Public

    async def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch a tool call to the appropriate handler.

        All handlers return a :class:`ToolResult` directly; the executor
        never raises on tool failure. Unknown tool names are reported as
        a :class:`ToolExecutionError` in the result.
        """
        try:
            if call.tool_name == LIST_FILES_TOOL.name:
                return await self._list_files(call)
            if call.tool_name == READ_FILE_TOOL.name:
                return await self._read_file(call)
            if call.tool_name == WRITE_FILE_TOOL.name:
                return await self._write_file(call)
            return self._error_result(
                call,
                FileToolError(f"Unknown file tool: {call.tool_name!r}"),
            )
        except FileToolError as exc:
            return self._error_result(call, exc)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning(
                "file_tool.unexpected  tool=%s  call_id=%s  error=%s",
                call.tool_name,
                call.call_id,
                type(exc).__name__,
            )
            return self._error_result(
                call,
                FileToolError(f"Tool execution failed: {type(exc).__name__}"),
            )

    # ------------------------------------------------------------------ list_files

    async def _list_files(self, call: ToolCall) -> ToolResult:
        directory = self._get_str(call.arguments, "directory", default="")
        # An empty directory argument means the workspace root.
        if directory:
            try:
                resolved_dir = self._workspace.resolve(directory)
            except (PathTraversalError, AbsolutePathError, SymlinkEscapeError) as exc:
                return self._error_result(
                    call,
                    FileToolError("Path is not allowed", path=exc.relative_path or ""),
                )
            except WorkspaceError as exc:
                return self._error_result(
                    call,
                    FileToolError("Path is not allowed", path=exc.relative_path or ""),
                )
        else:
            resolved_dir = self._workspace._canonical_root  # noqa: SLF001
            directory = ""

        try:
            stat_result = resolved_dir.stat()  # follow symlinks for listing
        except FileNotFoundError:
            return self._error_result(
                call,
                FileNotFoundError("Directory not found", path=directory),
            )
        except PermissionError:
            return self._error_result(
                call,
                PermissionDeniedError("Directory not accessible", path=directory),
            )
        except OSError as exc:
            return self._error_result(
                call,
                FileToolError(
                    f"Directory cannot be read: {type(exc).__name__}",
                    path=directory,
                ),
            )

        if not self._is_directory(resolved_dir):
            return self._error_result(
                call,
                FileNotFoundError("Not a directory", path=directory),
            )

        try:
            entries = list(resolved_dir.iterdir())
        except PermissionError:
            return self._error_result(
                call,
                PermissionDeniedError("Directory not accessible", path=directory),
            )
        except OSError as exc:
            return self._error_result(
                call,
                FileToolError(
                    f"Directory cannot be listed: {type(exc).__name__}",
                    path=directory,
                ),
            )

        if len(entries) > _MAX_LIST_ENTRIES:
            return self._error_result(
                call,
                FileToolError(
                    f"Directory has too many entries (>{_MAX_LIST_ENTRIES})",
                    path=directory,
                ),
            )

        items: list[FileListItem] = []
        for entry in entries:
            try:
                items.append(self._make_list_item(resolved_dir, entry, directory))
            except FileNotFoundError:
                # Entry was deleted between iterdir and stat. Skip it.
                continue
            except PermissionError:
                # Surface unreadable entries as items with safe metadata
                # so the caller can see they exist.
                items.append(
                    FileListItem(
                        name=entry.name,
                        path=self._join_relative(directory, entry.name),
                        is_file=False,
                        size_bytes=0,
                        mime_hint="application/octet-stream",
                        is_text=False,
                    )
                )
            except OSError:
                continue

        # Sort by name for deterministic output.
        items.sort(key=lambda it: it.name.lower())
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=[_dataclass_to_dict(it) for it in items],
            error=False,
            error_message="",
            latency_ms=0.0,
        )

    def _make_list_item(
        self,
        parent_dir: Path,
        entry: Path,
        request_dir: str,
    ) -> FileListItem:
        try:
            lst = entry.lstat()
        except OSError as exc:
            raise FileNotFoundError(
                f"Entry not accessible: {type(exc).__name__}",
                path=self._join_relative(request_dir, entry.name),
            ) from exc

        is_link = _is_symlink_like(lst)
        if is_link:
            # Resolve the symlink to determine if it points to a file
            # or directory. If the target is outside the workspace,
            # surface the entry as a symlink with a safe hint.
            try:
                real = entry.resolve(strict=False)
                if not _is_path_relative_to(real, parent_dir):
                    # Symlink escapes the workspace — surface as a
                    # non-file entry but still include the safe metadata.
                    return FileListItem(
                        name=entry.name,
                        path=self._join_relative(request_dir, entry.name),
                        is_file=False,
                        size_bytes=0,
                        mime_hint="inode/symlink",
                        is_text=False,
                    )
                target_stat = real.stat()
                is_dir = _is_directory_like(target_stat)
                size = target_stat.st_size
            except OSError:
                return FileListItem(
                    name=entry.name,
                    path=self._join_relative(request_dir, entry.name),
                    is_file=False,
                    size_bytes=0,
                    mime_hint="inode/symlink",
                    is_text=False,
                )
        else:
            try:
                target_stat = entry.stat()
            except OSError:
                raise
            is_dir = _is_directory_like(target_stat)
            size = target_stat.st_size

        ext = entry.suffix.lower()
        is_text = (not is_dir) and (ext in _SUPPORTED_TEXT_EXTENSIONS or ext == "")
        mime_hint = "text/plain" if is_text else "application/octet-stream"
        if is_dir:
            mime_hint = "inode/directory"
            is_text = False
            size = 0

        return FileListItem(
            name=entry.name,
            path=self._join_relative(request_dir, entry.name),
            is_file=(not is_dir) and (not is_link),
            size_bytes=size,
            mime_hint=mime_hint,
            is_text=is_text,
        )

    # ------------------------------------------------------------------ read_file

    async def _read_file(self, call: ToolCall) -> ToolResult:
        path = self._get_str(call.arguments, "path", default=None)
        if not path:
            return self._error_result(
                call,
                FileToolError("Missing required argument: path", path=""),
            )

        try:
            resolved = self._workspace.resolve(path)
        except (PathTraversalError, AbsolutePathError, SymlinkEscapeError) as exc:
            return self._error_result(
                call,
                FileToolError("Path is not allowed", path=exc.relative_path or ""),
            )
        except WorkspaceError as exc:
            return self._error_result(
                call,
                FileToolError("Path is not allowed", path=exc.relative_path or ""),
            )

        # Pre-flight checks before opening the file.
        ext = resolved.suffix.lower()
        if ext in _KNOWN_BINARY_EXTENSIONS:
            return self._error_result(
                call,
                UnsupportedFileError("File is not a supported text format", path=path),
            )
        if ext and ext not in _SUPPORTED_TEXT_EXTENSIONS:
            return self._error_result(
                call,
                UnsupportedFileError(
                    f"File extension {ext!r} is not supported",
                    path=path,
                ),
            )

        try:
            stat_result = resolved.stat()
        except FileNotFoundError:
            return self._error_result(
                call,
                FileNotFoundError("File not found", path=path),
            )
        except PermissionError:
            return self._error_result(
                call,
                PermissionDeniedError("File not accessible", path=path),
            )
        except OSError as exc:
            return self._error_result(
                call,
                FileToolError(
                    f"File cannot be read: {type(exc).__name__}",
                    path=path,
                ),
            )

        if _is_directory_like(stat_result):
            return self._error_result(
                call,
                UnsupportedFileError("Path is a directory, not a file", path=path),
            )

        if stat_result.st_size > _MAX_READ_BYTES:
            return self._error_result(
                call,
                UnsupportedFileError(
                    f"File is too large to read (>{_MAX_READ_BYTES} bytes)",
                    path=path,
                ),
            )

        # Read the file. We open in binary mode first and sniff for
        # binary content; only decode to text if it looks safe.
        try:
            with open(resolved, "rb") as fh:
                raw = fh.read(_MAX_READ_BYTES + 1)
        except PermissionError:
            return self._error_result(
                call,
                PermissionDeniedError("File not accessible", path=path),
            )
        except OSError as exc:
            return self._error_result(
                call,
                FileToolError(
                    f"File cannot be read: {type(exc).__name__}",
                    path=path,
                ),
            )

        if len(raw) > _MAX_READ_BYTES:
            return self._error_result(
                call,
                UnsupportedFileError(
                    f"File is too large to read (>{_MAX_READ_BYTES} bytes)",
                    path=path,
                ),
            )

        if _looks_like_binary(raw[:_BINARY_SNIFF_BYTES]):
            return self._error_result(
                call,
                UnsupportedFileError("File appears to be binary", path=path),
            )

        try:
            text = raw.decode(_TEXT_ENCODING, errors="strict")
        except UnicodeDecodeError:
            return self._error_result(
                call,
                UnsupportedFileError(
                    f"File is not valid {_TEXT_ENCODING} text",
                    path=path,
                ),
            )

        result = ReadResult(
            path=path,
            content=text,
            size_bytes=len(raw),
            encoding=_TEXT_ENCODING,
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=_dataclass_to_dict(result),
            error=False,
            error_message="",
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------ write_file

    async def _write_file(self, call: ToolCall) -> ToolResult:
        path = self._get_str(call.arguments, "path", default=None)
        content = self._get_content(call.arguments)
        create_dirs = self._get_bool(
            call.arguments, "create_directories", default=None
        )

        if not path:
            return self._error_result(
                call,
                FileToolError("Missing required argument: path", path=""),
            )
        if content is None:
            return self._error_result(
                call,
                FileToolError("Missing required argument: content", path=path),
            )

        # Validate size BEFORE resolving the path, so we don't touch
        # the filesystem for obviously invalid input.
        encoded = content.encode(_TEXT_ENCODING, errors="strict")
        if len(encoded) > _MAX_WRITE_BYTES:
            return self._error_result(
                call,
                UnsupportedFileError(
                    f"Content is too large to write (>{_MAX_WRITE_BYTES} bytes)",
                    path=path,
                ),
            )

        try:
            resolved = self._workspace.resolve(path)
        except (PathTraversalError, AbsolutePathError, SymlinkEscapeError) as exc:
            return self._error_result(
                call,
                FileToolError("Path is not allowed", path=exc.relative_path or ""),
            )
        except WorkspaceError as exc:
            return self._error_result(
                call,
                FileToolError("Path is not allowed", path=exc.relative_path or ""),
            )

        allow_create = (
            self._workspace.allow_create_dirs
            if create_dirs is None
            else bool(create_dirs) and self._workspace.allow_create_dirs
        )

        parent = resolved.parent
        created_dirs: list[str] = []
        if not parent.exists():
            if not allow_create:
                return self._error_result(
                    call,
                    FileNotFoundError(
                        "Parent directory does not exist and creation is not allowed",
                        path=path,
                    ),
                )
            try:
                self._create_parents(parent, created_dirs)
            except PermissionError:
                return self._error_result(
                    call,
                    PermissionDeniedError("Parent directory not accessible", path=path),
                )
            except OSError as exc:
                return self._error_result(
                    call,
                    FileToolError(
                        f"Parent directory cannot be created: {type(exc).__name__}",
                        path=path,
                    ),
                )

        # Write atomically: write to a temp file in the same directory,
        # then rename. The rename is atomic on POSIX and Windows for
        # files in the same directory.
        try:
            self._atomic_write(resolved, encoded)
        except PermissionError:
            return self._error_result(
                call,
                PermissionDeniedError("File not writable", path=path),
            )
        except IsADirectoryError:
            return self._error_result(
                call,
                FileToolError("Path is an existing directory", path=path),
            )
        except OSError as exc:
            return self._error_result(
                call,
                FileToolError(
                    f"File cannot be written: {type(exc).__name__}",
                    path=path,
                ),
            )

        result = WriteResult(
            path=path,
            size_bytes=len(encoded),
            created_directories=tuple(created_dirs),
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=_dataclass_to_dict(result),
            error=False,
            error_message="",
            latency_ms=0.0,
        )

    def _create_parents(self, parent: Path, created: list[str]) -> None:
        """Create parent directories up to (but not including) the workspace root.

        Stops at the workspace root: we do NOT create directories above the
        workspace. ``parent`` must already be inside the workspace (the
        path was validated by ``Workspace.resolve`` before this call).
        """
        workspace_root = self._workspace._canonical_root  # noqa: SLF001
        # Walk from the deepest missing directory up to the workspace
        # root. We use ``mkdir(parents=True)`` for the actual creation
        # but record the paths we created for the response.
        try:
            parent.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # A race: someone else created the directory between our
            # check and our create. That's fine.
            return

        # Walk back up to the workspace root, recording each created
        # directory in workspace-relative form.
        current = parent
        while current != workspace_root and workspace_root in current.parents:
            rel = self._workspace.relative_to(current)
            if rel:
                created.append(rel)
            if current.parent == current:
                break
            current = current.parent

    def _atomic_write(self, target: Path, data: bytes) -> None:
        """Write ``data`` to ``target`` atomically.

        The strategy is: write to a temp file in the same directory,
        fsync, then rename over the target. This ensures the target is
        never observed in a partially-written state.
        """
        import tempfile

        target_dir = target.parent
        # ``delete=False`` so we can fsync before close on Windows.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".write_",
            suffix=".tmp",
            dir=str(target_dir),
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # fsync may not be supported on all filesystems.
                    # We still proceed; the write is at least visible.
                    pass
            os.replace(tmp_path, target)
        except Exception:
            # Best-effort cleanup of the temp file.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ Helpers

    def _error_result(self, call: ToolCall, exc: FileToolError) -> ToolResult:
        """Build a ToolResult that represents a tool failure."""
        # The error_message is safe: it is the exception's message, which
        # does not include absolute host paths. We do NOT log the
        # arguments here.
        _logger.info(
            "file_tool.fail  tool=%s  call_id=%s  category=%s",
            call.tool_name,
            call.call_id,
            type(exc).__name__,
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=None,
            error=True,
            error_message=f"{type(exc).__name__}: {exc}"[:300],
            latency_ms=0.0,
        )

    @staticmethod
    def _get_str(
        args: Mapping[str, Any],
        key: str,
        default: Optional[str],
    ) -> Optional[str]:
        value = args.get(key, default)
        if value is None:
            return default
        if isinstance(value, str):
            return value
        # Coerce other types (int, etc.) to str. This is intentionally
        # permissive at the call-site level; the workspace rejects
        # anything that doesn't look like a path.
        return str(value)

    @staticmethod
    def _get_bool(
        args: Mapping[str, Any],
        key: str,
        default: Optional[bool],
    ) -> Optional[bool]:
        if key not in args:
            return default
        value = args[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
        return default

    @staticmethod
    def _get_content(args: Mapping[str, Any]) -> Optional[str]:
        if "content" not in args:
            return None
        value = args["content"]
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode(_TEXT_ENCODING)
            except UnicodeDecodeError:
                return None
        return None

    @staticmethod
    def _join_relative(parent: str, name: str) -> str:
        if not parent:
            return name
        # Normalise: forward slashes, strip trailing slash.
        p = parent.replace("\\", "/").rstrip("/")
        return f"{p}/{name}"

    @staticmethod
    def _is_directory(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def create_file_tool_executor(workspace: Workspace) -> FileToolExecutor:
    """Build a :class:`FileToolExecutor` for the given workspace.

    Equivalent to ``FileToolExecutor(workspace)``; provided as a
    convenience so callers can use a single import.
    """
    return FileToolExecutor(workspace)


# ---------------------------------------------------------------------------
# Module helpers (private)
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass to a plain dict.

    Used for tool outputs so the result is JSON-friendly without pulling
    in a framework.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _looks_like_binary(data: bytes) -> bool:
    """Heuristic: does the given byte string look like binary content?

    A byte sequence is considered binary if it contains a NUL byte, or
    if it cannot be decoded as UTF-8. A small NUL-free prefix that
    successfully decodes as UTF-8 is treated as text.

    This is conservative: a file that *could* be valid UTF-8 is treated
    as text, even if it might actually be a binary file in a compatible
    encoding. False positives (refusing a binary file that happens to
    look like UTF-8) are less harmful than false negatives (reading a
    binary file as text and returning garbage to the LLM).
    """
    if not data:
        return False
    if b"\x00" in data:
        return True
    try:
        data.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def _is_symlink_like(stat_result) -> bool:
    """Return True if the stat result indicates a symlink or reparse point."""
    import sys as _sys

    if _sys.platform == "win32":
        # FILE_ATTRIBUTE_REPARSE_POINT == 0x400
        attrs = getattr(stat_result, "st_file_attributes", 0)
        if attrs & 0x400:
            return True
    # POSIX: S_ISLNK checks the file-type bits.
    return (stat_result.st_mode & 0o170000) == 0o120000


def _is_directory_like(stat_result) -> bool:
    """Return True if the stat result indicates a directory."""
    return (stat_result.st_mode & 0o170000) == 0o040000


def _is_path_relative_to(path: Path, base: Path) -> bool:
    """Portable ``is_relative_to`` (Python 3.9+ has it natively)."""
    if hasattr(path, "is_relative_to"):
        try:
            return path.is_relative_to(base)
        except ValueError:
            return False
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
