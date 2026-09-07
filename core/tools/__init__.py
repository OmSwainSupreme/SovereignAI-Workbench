"""SovereignAI Workbench — File & Document Tools (Phase 5A, 5E).

Secure file system tools and document generation that operate within an
explicitly configured workspace.

Example usage::

    from core.tools.workspace import Workspace
    from core.tools.file_tools import create_file_tool_executor
    from core.tools.document_tools import create_document_tool_executor

    workspace = Workspace(root_path="/data/workspace")
    file_executor = create_file_tool_executor(workspace)
    doc_executor = create_document_tool_executor(workspace)
    # Register with ToolRegistry...
"""
from core.tools.workspace import Workspace, WorkspaceError, PathTraversalError
from core.tools.types import (
    FileListItem,
    ReadResult,
    WriteResult,
    FileToolError,
    FileNotFoundError,
    UnsupportedFileError,
    PermissionDeniedError,
)
from core.tools.file_tools import (
    FileToolExecutor,
    create_file_tool_executor,
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
)
from core.tools.document_tools import (
    DocumentToolExecutor,
    create_document_tool_executor,
    CREATE_DOCUMENT_TOOL,
    DocumentToolError,
    InvalidContentError,
    DocumentTooLargeError,
    LibraryError,
)
from core.tools.registry import register_file_tools, register_document_tools

__all__ = [
    # workspace
    "Workspace",
    "WorkspaceError",
    "PathTraversalError",
    # file types
    "FileListItem",
    "ReadResult",
    "WriteResult",
    "FileToolError",
    "UnsupportedFileError",
    "FileNotFoundError",
    "PermissionDeniedError",
    # file tools
    "FileToolExecutor",
    "create_file_tool_executor",
    "LIST_FILES_TOOL",
    "READ_FILE_TOOL",
    "WRITE_FILE_TOOL",
    # document types
    "DocumentToolError",
    "InvalidContentError",
    "DocumentTooLargeError",
    "LibraryError",
    # document tools
    "DocumentToolExecutor",
    "create_document_tool_executor",
    "CREATE_DOCUMENT_TOOL",
    # registry
    "register_file_tools",
    "register_document_tools",
]
