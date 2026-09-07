"""Framework-agnostic data types for File Tools (Phase 5A).

These types are deliberately plain dataclasses — they must not depend on
FastAPI, Pydantic, or any other framework.

This mirrors the design convention of :mod:`core.agent.types`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileListItem:
    """A single entry returned by :func:`list_files`.

    All paths are *relative* to the workspace root — the tool NEVER exposes
    absolute host paths in its output.
    """

    name: str
    """The entry's basename (filename or directory name)."""

    path: str
    """Path relative to the workspace root, using forward slashes."""

    is_file: bool
    """True if the entry is a regular file, False if directory or other."""

    size_bytes: int
    """File size in bytes; 0 for directories."""

    mime_hint: str
    """A conservative MIME-type hint (e.g. ``"text/plain"`` or
    ``"application/octet-stream"`` for unsupported files)."""

    is_text: bool
    """True if the file appears to be a supported text file."""


@dataclass(frozen=True)
class ReadResult:
    """The result of a successful :func:`read_file` call."""

    path: str
    """Path relative to the workspace root."""

    content: str
    """The file's text content (already decoded with the configured encoding)."""

    size_bytes: int
    """Size of the file in bytes."""

    encoding: str
    """Encoding used to read the file (e.g. ``"utf-8"``)."""


@dataclass(frozen=True)
class WriteResult:
    """The result of a successful :func:`write_file` call."""

    path: str
    """Path relative to the workspace root."""

    size_bytes: int
    """Size of the written content in bytes."""

    created_directories: tuple[str, ...] = field(default_factory=tuple)
    """Directories created during this write, as workspace-relative paths.

    Empty tuple if no new directories were created (target directory already
    existed or write is at the workspace root).
    """


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class FileToolError(Exception):
    """Base class for all file-tool errors.

    The error categories mirror what a future Policy Engine will need to
    distinguish between different denial reasons. Each subclass maps to one
    diagnostic category:

    * :class:`FileNotFoundError` — entry does not exist.
    * :class:`UnsupportedFileError` — entry exists but is not supported
      (binary, wrong extension, too large, etc.).
    * :class:`PermissionDeniedError` — entry exists but cannot be read
      or written under the current policy.
    * :class:`FileToolError` (base) — generic tool execution failure
      (e.g. I/O error, path validation failure that does not fit the
      above categories).

    Tool errors NEVER include the absolute host path in the message; they
    carry only safe, workspace-relative information.
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
    ) -> None:
        self.path = path
        super().__init__(message)


class FileNotFoundError(FileToolError):
    """The requested file or directory does not exist inside the workspace."""


class UnsupportedFileError(FileToolError):
    """The file exists but is not supported by this tool (e.g. binary)."""


class PermissionDeniedError(FileToolError):
    """Access to the path is denied by the workspace policy."""
