"""File-related routes for the v1 API.

Exposes endpoints for workspace file operations: list, upload, download.
All operations go through the workspace-validated file tools and policy engine.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.core.config import settings
from core.agent.types import ToolCall
from core.security.policy_engine import get_policy_engine
from core.tools.file_tools import create_file_tool_executor, LIST_FILES_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL
from core.tools.workspace import Workspace

logger = logging.getLogger("sovereign-ai.api.files")


def _iso_utc(ts: float) -> str:
    """Format a UTC POSIX timestamp as an ISO-8601 string for the frontend."""
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

router = APIRouter(prefix="/files", tags=["files"])


def get_workspace() -> Workspace:
    """Return a workspace instance for file operations."""
    workspace_path = settings.workspace_path or "workspace"
    return Workspace(workspace_path)


def get_file_tool_executor() -> object:
    """Return a file tool executor for the workspace."""
    workspace = get_workspace()
    return create_file_tool_executor(workspace)


class FileEntry(BaseModel):
    id: str
    name: str
    kind: str  # "file" or "folder"
    origin: str  # "user_upload" | "generated" | "unknown"
    sizeBytes: Optional[int] = None
    updatedAt: str
    contentType: Optional[str] = None
    handle: str
    downloadable: bool


class FileListResponse(BaseModel):
    files: List[FileEntry]


class UploadResponse(BaseModel):
    message: str
    uploaded: List[str]


@router.get("", response_model=FileListResponse)
async def list_files() -> FileListResponse:
    """List all files and folders in the workspace."""
    executor = get_file_tool_executor()
    workspace = get_workspace()
    policy_engine = get_policy_engine()

    # Create a tool call for list_files
    tool_call = ToolCall(
        call_id="list_files_call",
        tool_name="list_files",
        arguments={},
    )

    # Check policy
    policy_decision = policy_engine.evaluate(tool_call)
    if not policy_decision.is_allowed():
        logger.warning("Policy denied list_files: %s", policy_decision.reason)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=policy_decision.reason,
        )

    # Execute the tool
    try:
        result = await executor.execute(tool_call)
        if result.error:
            logger.error("Error listing files: %s", result.error_message)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list files",
            )

        # The result.output is the list_files tool's FileListItem dicts:
        # {name, path, is_file, size_bytes, mime_hint, is_text}. Map them into
        # the frontend-facing FileEntry shape. The workspace-relative `path` is
        # used as the opaque handle — it is echoed back for downloads and never
        # exposes a host filesystem path.
        files = result.output  # type: ignore
        file_entries: List[FileEntry] = []
        for item in files:
            rel_path = item.get("path") or item.get("name") or ""
            is_file = bool(item.get("is_file"))
            try:
                mtime = workspace.resolve(rel_path).stat().st_mtime
            except Exception:  # noqa: BLE001 - stat failure is non-fatal
                mtime = 0.0
            file_entries.append(
                FileEntry(
                    id=rel_path,
                    name=item.get("name") or rel_path,
                    kind="file" if is_file else "folder",
                    origin="user_upload" if is_file else "unknown",
                    sizeBytes=item.get("size_bytes"),
                    updatedAt=_iso_utc(mtime),
                    contentType=item.get("mime_hint"),
                    handle=rel_path,
                    downloadable=is_file,
                )
            )
        return FileListResponse(files=file_entries)
    except Exception as exc:
        logger.exception("Unexpected error listing files: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
) -> UploadResponse:
    """Upload one or more files to the workspace.

    Files are written through the same workspace-validated ``write_file`` tool
    as the agent uses, so every upload is subject to path validation, policy
    checks, and the tool's format/size limits. Unsupported content (binary,
    non-UTF-8, oversized) is rejected with a clean client error rather than a
    server error — the fault is the caller's file, not the backend.
    """
    executor = get_file_tool_executor()
    workspace = get_workspace()
    policy_engine = get_policy_engine()

    IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})
    MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MiB limit

    uploaded_names = []
    for upload_file in files:
        filename = upload_file.filename or "unnamed"
        try:
            resolved_path = workspace.resolve(filename)
        except Exception as exc:
            logger.warning("Upload path rejected for %s: %s", filename, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Upload rejected for {filename}: invalid or unsafe path",
            )

        content = await upload_file.read()
        await upload_file.seek(0)

        if len(content) > MAX_FILE_BYTES:
            logger.warning("Upload rejected (file too large): %s (%d bytes)", filename, len(content))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {filename} exceeds maximum size of 50 MiB.",
            )

        # Policy evaluation on write_file capability
        tool_call = ToolCall(
            call_id=f"write_file_call_{filename}",
            tool_name="write_file",
            arguments={"path": filename},
        )
        policy_decision = policy_engine.evaluate(tool_call)
        if not policy_decision.is_allowed():
            logger.warning("Policy denied upload for %s: %s", filename, policy_decision.reason)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Policy denied upload for {filename}: {policy_decision.reason}",
            )

        # Check if this is an image file
        file_ext = Path(filename).suffix.lower()
        if file_ext in IMAGE_EXTENSIONS:
            try:
                resolved_path.write_bytes(content)
                uploaded_names.append(filename)
                logger.info("Successfully uploaded binary image: %s (%d bytes)", filename, len(content))
                continue
            except Exception as exc:
                logger.exception("Failed to write image file %s: %s", filename, exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error writing image {filename}",
                )

        # Text file path: must be valid UTF-8
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Upload refused (non-UTF-8 content): %s", filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {filename} is not supported text content (UTF-8).",
            )

        tool_call.arguments["content"] = text_content
        try:
            result = await executor.execute(tool_call)
            if result.error:
                logger.error("Error writing file %s: %s", filename, result.error_message)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Upload rejected for {filename}: {result.error_message}",
                )
            uploaded_names.append(filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error writing file %s: %s", filename, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error for file {filename}",
            )

    return UploadResponse(
        message=f"Successfully uploaded {len(uploaded_names)} file(s)",
        uploaded=uploaded_names,
    )


@router.get("/{handle}")
async def download_file(handle: str):
    """Download a file from the workspace by its handle."""
    executor = get_file_tool_executor()
    workspace = get_workspace()
    policy_engine = get_policy_engine()

    # First, we need to get the file metadata to know the path and content type.
    # We can use the list_files tool and find the file with the given handle.
    # Alternatively, we can have a tool to get file info by handle, but we don't have one.
    # We'll list files and find the matching handle.
    list_tool_call = ToolCall(
        call_id="list_files_for_download",
        tool_name="list_files",
        arguments={},
    )
    policy_decision = policy_engine.evaluate(list_tool_call)
    if not policy_decision.is_allowed():
        logger.warning("Policy denied list_files for download: %s", policy_decision.reason)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=policy_decision.reason,
        )

    list_result = await executor.execute(list_tool_call)
    if list_result.error:
        logger.error("Error listing files for download: %s", list_result.error_message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files",
        )

    files = list_result.output  # type: ignore
    file_info = None
    for f in files:
        # The handle echoed from the list endpoint is the workspace-relative
        # path; match on it (fall back to the basename for robustness).
        if f.get("path") == handle or f.get("name") == handle:
            file_info = f
            break

    if not file_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with handle {handle} not found",
        )

    # Now, read the file using the read_file tool or workspace for images
    target_path = file_info.get("path") or file_info.get("name")
    read_tool_call = ToolCall(
        call_id=f"read_file_{handle}",
        tool_name="read_file",
        arguments={
            "path": target_path,
        },
    )
    policy_decision = policy_engine.evaluate(read_tool_call)
    if not policy_decision.is_allowed():
        logger.warning("Policy denied read_file for %s: %s", handle, policy_decision.reason)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=policy_decision.reason,
        )

    file_ext = Path(target_path).suffix.lower()
    IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})

    if file_ext in IMAGE_EXTENSIONS:
        try:
            resolved_path = workspace.resolve(target_path)
            content_bytes = resolved_path.read_bytes()
            media_type = file_info.get("mime_hint") or f"image/{file_ext.lstrip('.')}"
            if file_ext in (".jpg", ".jpeg"):
                media_type = "image/jpeg"
            return StreamingResponse(
                iter([content_bytes]),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"inline; filename={file_info['name']}"
                },
            )
        except Exception as exc:
            logger.exception("Failed to read image file %s: %s", handle, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to read image file",
            )

    read_result = await executor.execute(read_tool_call)
    if read_result.error:
        logger.error("Error reading file %s: %s", handle, read_result.error_message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read file",
        )

    # read_file returns a ReadResult dict ({path, content, size_bytes, encoding}).
    # Extract the text content; never ship the whole dict as the body.
    content_data = read_result.output  # type: ignore
    if isinstance(content_data, dict):
        content = content_data.get("content") or ""
    else:
        content = content_data or ""

    # Return as a streaming response. The list tool reports the MIME hint as
    # `mime_hint`; use it so the client gets the right content type.
    media_type = file_info.get("mime_hint") or "application/octet-stream"
    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={file_info['name']}"
        },
    )