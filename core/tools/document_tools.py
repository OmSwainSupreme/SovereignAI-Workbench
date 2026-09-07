"""Document generation tools (Phase 5E) — create_document.

This tool operates ONLY inside an explicitly configured :class:`Workspace`.
All path validation is delegated to the workspace; the tool itself does
not perform any path canonicalisation or traversal checks.

Security summary
----------------

* Every tool argument is passed through :meth:`Workspace.resolve` first.
* The executor never touches paths outside the workspace.
* Uses atomic/safe file creation (write to temp file then rename).
* Enforces reasonable output-size limits (configurable).
* All errors are returned as :class:`ToolResult` with ``error=True`` and
  a non-sensitive ``error_message``; the tool never raises to the caller.

Tool contracts
--------------

create_document(path, title, content, metadata=None)
    Generate a DOCX document at ``path`` (relative to workspace) with the given
    title, structured content, and optional metadata.
    ``content`` is a list of document elements (paragraphs, headings, lists, tables).
    Returns :class:`WriteResult` with the size of the written document in bytes.
    Intermediate directories are created only if explicitly allowed by the workspace.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from core.agent.interfaces import ToolDefinition, ToolExecutor
from core.agent.types import ToolCall, ToolResult
from core.tools.types import (
    FileToolError,
    FileNotFoundError,
    PermissionDeniedError,
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


_logger = logging.getLogger("sovereign-ai.tools.document")

# Maximum output file size (in bytes). Conservative guard.
_MAX_OUTPUT_BYTES = 20 * 1024 * 1024  # 20 MiB

# Maximum number of content elements in a single document.
_MAX_CONTENT_ELEMENTS = 500

# Maximum number of elements in a single list.
_MAX_LIST_ITEMS = 200

# Maximum rows/columns in a table.
_MAX_TABLE_ROWS = 100
_MAX_TABLE_COLS = 50

# Maximum text length for a single text node.
_MAX_TEXT_LENGTH = 50_000


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


CREATE_DOCUMENT_TOOL = ToolDefinition(
    name="create_document",
    description=(
        "Generate a DOCX document inside the workspace. "
        "Supports structured content: headings, paragraphs, bullet lists, "
        "numbered lists, and tables. Returns the workspace-relative path "
        "and size of the written file."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative output path for the .docx file. "
                    "Must end with .docx."
                ),
            },
            "title": {
                "type": "string",
                "description": "Document title (used as heading and metadata).",
            },
            "content": {
                "type": "array",
                "description": (
                    "Ordered list of content elements. Each element is an "
                    "object with a 'type' key: 'heading', 'paragraph', "
                    "'bullet_list', 'numbered_list', or 'table'."
                ),
                "items": {
                    "type": "object",
                },
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Optional document metadata: author, subject, keywords."
                ),
            },
        },
        "required": ["path", "title", "content"],
    },
    output_description=(
        "A {path, size_bytes, created_directories} object on success."
    ),
    capability="document_generation",
)


# ---------------------------------------------------------------------------
# Content element types
# ---------------------------------------------------------------------------

_VALID_ELEMENT_TYPES = frozenset({
    "heading",
    "paragraph",
    "bullet_list",
    "numbered_list",
    "table",
})

_VALID_HEADING_LEVELS = frozenset({1, 2, 3, 4, 5, 6})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DocumentToolError(FileToolError):
    """Base error for document tool failures."""


class InvalidContentError(DocumentToolError):
    """The document content structure is invalid."""


class DocumentTooLargeError(DocumentToolError):
    """The generated document exceeds the output size limit."""


class LibraryError(DocumentToolError):
    """The underlying document library raised an error."""


# ---------------------------------------------------------------------------
# DocumentToolExecutor
# ---------------------------------------------------------------------------


class DocumentToolExecutor(ToolExecutor):
    """A :class:`ToolExecutor` that creates DOCX documents inside a workspace.

    The executor is bound to a single :class:`Workspace`.  All path
    validation is delegated to the workspace; the executor only builds
    the document and writes it atomically.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    # ------------------------------------------------------------------ Public

    async def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch a tool call to the appropriate handler."""
        try:
            if call.tool_name == CREATE_DOCUMENT_TOOL.name:
                return await self._create_document(call)
            return self._error_result(
                call,
                DocumentToolError(f"Unknown document tool: {call.tool_name!r}"),
            )
        except DocumentToolError as exc:
            return self._error_result(call, exc)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning(
                "doc_tool.unexpected  tool=%s  call_id=%s  error=%s",
                call.tool_name,
                call.call_id,
                type(exc).__name__,
            )
            return self._error_result(
                call,
                DocumentToolError(
                    f"Document generation failed: {type(exc).__name__}"
                ),
            )

    # ------------------------------------------------------------------ create_document

    async def _create_document(self, call: ToolCall) -> ToolResult:
        path = self._get_str(call.arguments, "path", default=None)
        title = self._get_str(call.arguments, "title", default=None)
        content = call.arguments.get("content")
        metadata = call.arguments.get("metadata")

        # --- Argument validation (before touching the filesystem) ---

        if not path:
            return self._error_result(
                call,
                DocumentToolError("Missing required argument: path", path=""),
            )
        if not title:
            return self._error_result(
                call,
                DocumentToolError("Missing required argument: title", path=path),
            )
        if content is None:
            return self._error_result(
                call,
                DocumentToolError("Missing required argument: content", path=path),
            )

        # Validate title length.
        if not isinstance(title, str) or not title.strip():
            return self._error_result(
                call,
                InvalidContentError("Title must be a non-empty string", path=path),
            )

        # Validate content structure.
        if not isinstance(content, list):
            return self._error_result(
                call,
                InvalidContentError(
                    "Content must be a list of elements", path=path
                ),
            )

        if len(content) > _MAX_CONTENT_ELEMENTS:
            return self._error_result(
                call,
                InvalidContentError(
                    f"Content has too many elements (>{_MAX_CONTENT_ELEMENTS})",
                    path=path,
                ),
            )

        content_err = self._validate_content(content)
        if content_err:
            return self._error_result(call, InvalidContentError(content_err, path=path))

        # Validate metadata if provided.
        if metadata is not None and not isinstance(metadata, dict):
            return self._error_result(
                call,
                InvalidContentError("Metadata must be an object", path=path),
            )

        # --- Path resolution ---

        try:
            resolved = self._workspace.resolve(path)
        except (PathTraversalError, AbsolutePathError, SymlinkEscapeError) as exc:
            return self._error_result(
                call,
                DocumentToolError("Path is not allowed", path=exc.relative_path or ""),
            )
        except WorkspaceError as exc:
            return self._error_result(
                call,
                DocumentToolError("Path is not allowed", path=exc.relative_path or ""),
            )

        # Ensure .docx extension.
        if resolved.suffix.lower() != ".docx":
            return self._error_result(
                call,
                InvalidContentError(
                    "Output path must end with .docx", path=path
                ),
            )

        # --- Ensure parent directory exists ---

        parent = resolved.parent
        created_dirs: list[str] = []
        if not parent.exists():
            if not self._workspace.allow_create_dirs:
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
                    DocumentToolError(
                        f"Parent directory cannot be created: {type(exc).__name__}",
                        path=path,
                    ),
                )

        # --- Generate the document ---

        try:
            docx_bytes = self._build_docx(title.strip(), content, metadata)
        except InvalidContentError as exc:
            return self._error_result(call, exc)
        except Exception as exc:
            _logger.warning(
                "doc_tool.build_error  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return self._error_result(
                call,
                LibraryError(
                    f"Document generation failed: {type(exc).__name__}"
                ),
            )

        # Size guard.
        if len(docx_bytes) > _MAX_OUTPUT_BYTES:
            return self._error_result(
                call,
                DocumentTooLargeError(
                    f"Generated document is too large (>{_MAX_OUTPUT_BYTES} bytes)"
                ),
            )

        # --- Pre-write checks ---

        # If the resolved path already exists as a directory, fail cleanly
        # rather than letting the OS error message leak through.
        if resolved.exists() and resolved.is_dir():
            return self._error_result(
                call,
                DocumentToolError("Path is an existing directory", path=path),
            )

        # --- Atomic write ---

        try:
            self._atomic_write(resolved, docx_bytes)
        except PermissionError:
            return self._error_result(
                call,
                PermissionDeniedError("File not writable", path=path),
            )
        except IsADirectoryError:
            return self._error_result(
                call,
                DocumentToolError("Path is an existing directory", path=path),
            )
        except OSError as exc:
            return self._error_result(
                call,
                DocumentToolError(
                    f"File cannot be written: {type(exc).__name__}",
                    path=path,
                ),
            )

        result = WriteResult(
            path=path,
            size_bytes=len(docx_bytes),
            created_directories=tuple(created_dirs),
        )
        _logger.info(
            "doc_tool.success  call_id=%s  path=%s  size=%d",
            call.call_id,
            path,
            len(docx_bytes),
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=_dataclass_to_dict(result),
            error=False,
            error_message="",
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------ DOCX builder

    def _build_docx(
        self,
        title: str,
        content: Sequence[dict[str, Any]],
        metadata: Optional[dict[str, Any]],
    ) -> bytes:
        """Build a DOCX document in memory and return the raw bytes.

        Uses python-docx. All validation has already happened.
        """
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()

        # --- Core properties / metadata ---
        if metadata:
            core = doc.core_properties
            if "author" in metadata:
                core.author = str(metadata["author"])[:255]
            if "subject" in metadata:
                core.subject = str(metadata["subject"])[:255]
            if "keywords" in metadata:
                core.keywords = str(metadata["keywords"])[:255]
            if "category" in metadata:
                core.category = str(metadata["category"])[:255]

        # --- Title as a heading ---
        doc.add_heading(title, level=0)

        # --- Content elements ---
        for element in content:
            etype = element.get("type", "")

            if etype == "heading":
                self._add_heading(doc, element)
            elif etype == "paragraph":
                self._add_paragraph(doc, element)
            elif etype == "bullet_list":
                self._add_bullet_list(doc, element)
            elif etype == "numbered_list":
                self._add_numbered_list(doc, element)
            elif etype == "table":
                self._add_table(doc, element)
            # Invalid types already filtered by _validate_content.

        # --- Serialize ---
        from io import BytesIO

        buf = BytesIO()
        doc.save(buf)
        return buf.getvalue()

    # ------------------------------------------------------------------ Element helpers

    @staticmethod
    def _get_text(element: dict[str, Any], key: str = "text") -> str:
        """Safely extract a text string from an element dict."""
        val = element.get(key, "")
        if not isinstance(val, str):
            val = str(val)
        # Truncate to limit.
        if len(val) > _MAX_TEXT_LENGTH:
            val = val[:_MAX_TEXT_LENGTH]
        return val

    @staticmethod
    def _add_heading(doc: Any, element: dict[str, Any]) -> None:
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        text = DocumentToolExecutor._get_text(element)
        level = element.get("level", 1)
        if not isinstance(level, int) or level not in _VALID_HEADING_LEVELS:
            level = 1
        doc.add_heading(text, level=level)

    @staticmethod
    def _add_paragraph(doc: Any, element: dict[str, Any]) -> None:
        text = DocumentToolExecutor._get_text(element)
        style = element.get("style")
        if style and isinstance(style, str):
            try:
                doc.add_paragraph(text, style=style)
                return
            except (KeyError, ValueError):
                pass
        doc.add_paragraph(text)

    @staticmethod
    def _add_bullet_list(doc: Any, element: dict[str, Any]) -> None:
        items = element.get("items", [])
        if not isinstance(items, list):
            return
        for item in items[:_MAX_LIST_ITEMS]:
            doc.add_paragraph(
                DocumentToolExecutor._get_text({"text": item}),
                style="List Bullet",
            )

    @staticmethod
    def _add_numbered_list(doc: Any, element: dict[str, Any]) -> None:
        items = element.get("items", [])
        if not isinstance(items, list):
            return
        for item in items[:_MAX_LIST_ITEMS]:
            doc.add_paragraph(
                DocumentToolExecutor._get_text({"text": item}),
                style="List Number",
            )

    @staticmethod
    def _add_table(doc: Any, element: dict[str, Any]) -> None:
        rows = element.get("rows", [])
        if not isinstance(rows, list) or not rows:
            return
        # Validate all rows have the same number of columns.
        col_count = None
        for row in rows[:_MAX_TABLE_ROWS]:
            if not isinstance(row, list):
                return
            if col_count is None:
                col_count = len(row)
                if col_count == 0 or col_count > _MAX_TABLE_COLS:
                    return
            else:
                # Pad or truncate to col_count.
                pass

        if col_count is None or col_count == 0:
            return

        table = doc.add_table(rows=0, cols=col_count)
        table.style = "Table Grid"

        for row_data in rows[:_MAX_TABLE_ROWS]:
            if not isinstance(row_data, list):
                continue
            row_cells = table.add_row().cells
            for i in range(col_count):
                cell_text = row_data[i] if i < len(row_data) else ""
                row_cells[i].text = DocumentToolExecutor._get_text(
                    {"text": cell_text}
                )

    # ------------------------------------------------------------------ Validation

    @staticmethod
    def _validate_content(content: Sequence[dict[str, Any]]) -> Optional[str]:
        """Validate the content list structure.  Returns an error string or None."""
        for i, element in enumerate(content):
            if not isinstance(element, dict):
                return f"Element at index {i} must be an object"
            etype = element.get("type", "")
            if etype not in _VALID_ELEMENT_TYPES:
                return (
                    f"Element at index {i} has invalid type {etype!r}; "
                    f"valid types: {sorted(_VALID_ELEMENT_TYPES)}"
                )
            if etype in ("heading", "paragraph"):
                text = element.get("text")
                if text is not None and not isinstance(text, str):
                    if not isinstance(text, (int, float)):
                        return f"Element at index {i}: text must be a string"
            elif etype in ("bullet_list", "numbered_list"):
                items = element.get("items")
                if items is not None:
                    if not isinstance(items, list):
                        return f"Element at index {i}: items must be a list"
                    if len(items) > _MAX_LIST_ITEMS:
                        return (
                            f"Element at index {i}: list has too many items "
                            f"(>{_MAX_LIST_ITEMS})"
                        )
            elif etype == "table":
                rows = element.get("rows")
                if rows is not None:
                    if not isinstance(rows, list):
                        return f"Element at index {i}: rows must be a list"
                    if len(rows) > _MAX_TABLE_ROWS:
                        return (
                            f"Element at index {i}: table has too many rows "
                            f"(>{_MAX_TABLE_ROWS})"
                        )
                    for j, row in enumerate(rows):
                        if not isinstance(row, list):
                            return (
                                f"Element at index {i}, row {j}: "
                                "row must be a list"
                            )
                        if len(row) > _MAX_TABLE_COLS:
                            return (
                                f"Element at index {i}, row {j}: "
                                f"too many columns (>{_MAX_TABLE_COLS})"
                            )
        return None

    # ------------------------------------------------------------------ Filesystem helpers

    def _create_parents(self, parent: Path, created: list[str]) -> None:
        """Create parent directories up to (but not including) the workspace root."""
        workspace_root = self._workspace._canonical_root  # noqa: SLF001
        try:
            parent.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return
        current = parent
        while current != workspace_root and workspace_root in current.parents:
            rel = self._workspace.relative_to(current)
            if rel:
                created.append(rel)
            if current.parent == current:
                break
            current = current.parent

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        """Write ``data`` to ``target`` atomically via temp file + rename."""
        target_dir = target.parent
        fd, tmp_path = tempfile.mkstemp(
            prefix=".docgen_",
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
                    pass
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ Helpers

    def _error_result(self, call: ToolCall, exc: DocumentToolError) -> ToolResult:
        """Build a ToolResult that represents a tool failure."""
        _logger.info(
            "doc_tool.fail  tool=%s  call_id=%s  category=%s",
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
        return str(value)


# ---------------------------------------------------------------------------
# Dataclass-to-dict helper (shared with file_tools)
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def create_document_tool_executor(workspace: Workspace) -> DocumentToolExecutor:
    """Build a :class:`DocumentToolExecutor` for the given workspace."""
    return DocumentToolExecutor(workspace)
