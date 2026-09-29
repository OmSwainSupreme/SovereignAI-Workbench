"""Vision-related routes for the v1 API.

Exposes endpoints for OCR and vision operations.
All operations go through the vision tools and policy engine.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from core.agent.types import ToolCall
from core.security.policy_engine import get_policy_engine
from core.tools.workspace import Workspace, WorkspaceError
from core.vision.image_loader import ImageLoader
from core.vision.ocr import OCRProvider
from core.vision.vision import VisionProvider
from core.vision.tools import (
    ANALYZE_IMAGE_TOOL,
    OCR_IMAGE_TOOL,
    _analyze_image_callable,
    _ocr_image_callable,
)

logger = logging.getLogger("sovereign-ai.api.vision")

router = APIRouter(prefix="/vision", tags=["vision"])


class OCRRequest(BaseModel):
    language: Optional[str] = Field(None, description="Optional language hint (e.g. 'en').")


class AnalyzeRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Optional natural-language instruction.")


class OCRResponse(BaseModel):
    image_id: str
    full_text: str
    confidence: float
    page_number: int
    page_count: int
    language: Optional[str] = None
    block_count: int
    blocks: list[dict]
    provider: str


class AnalyzeResponse(BaseModel):
    image_id: str
    description: str
    confidence: float
    tags: list[str]
    regions: list[dict]
    provider: str


def get_workspace() -> Workspace:
    """Return a workspace instance for file operations."""
    workspace_path = settings.workspace_path or "workspace"
    return Workspace(workspace_path)


def get_image_loader(workspace: Workspace) -> ImageLoader:
    """Return an image loader for the workspace."""
    return ImageLoader(workspace)


def get_ocr_provider() -> OCRProvider:
    """Return an OCR provider configured with the local vision model."""
    from core.vision.ollama_ocr import OllamaOCRProvider
    return OllamaOCRProvider(
        base_url=settings.llm_ollama_base_url,
        default_model="qwen2.5vl:3b",
        request_timeout_seconds=settings.llm_ollama_request_timeout_seconds,
    )


def get_vision_provider() -> VisionProvider:
    """Return a vision provider configured with the local vision model."""
    from core.vision.ollama_vision import OllamaVisionProvider
    return OllamaVisionProvider(
        base_url=settings.llm_ollama_base_url,
        default_model="qwen2.5vl:3b",
        request_timeout_seconds=settings.llm_ollama_request_timeout_seconds,
    )


@router.post("/ocr", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    request: OCRRequest = Depends(),
) -> OCRResponse:
    """Run OCR on an uploaded image."""
    workspace = get_workspace()
    image_loader = get_image_loader(workspace)
    ocr_provider = get_ocr_provider()
    policy_engine = get_policy_engine()

    filename = file.filename or "uploaded_image.png"

    # Validate workspace path security (path traversal, absolute path, etc.)
    try:
        resolved = workspace.resolve(filename)
    except WorkspaceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image filename: {exc}",
        )

    content = await file.read()

    # Check policy for write_file / image upload
    tool_call_write = ToolCall(
        call_id=f"write_file_{filename}",
        tool_name="write_file",
        arguments={"path": filename},
    )
    if policy_engine is not None:
        policy_decision = policy_engine.evaluate(tool_call_write)
        if not policy_decision.is_allowed():
            logger.warning("Policy denied image write for %s: %s", filename, policy_decision.reason)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Policy denied upload for {filename}: {policy_decision.reason}",
            )

    # Save raw binary bytes to workspace
    try:
        resolved.write_bytes(content)
    except OSError as exc:
        logger.error("Error writing image file %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write image file {filename}",
        )

    # Create a tool call for ocr_image
    ocr_tool_call = ToolCall(
        call_id=f"ocr_image_{filename}",
        tool_name="ocr_image",
        arguments={
            "path": filename,
            "language": request.language or "",
        },
    )

    # Check policy for ocr_image
    if policy_engine is not None:
        policy_decision = policy_engine.evaluate(ocr_tool_call)
        if not policy_decision.is_allowed():
            logger.warning("Policy denied ocr_image for %s: %s", filename, policy_decision.reason)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Policy denied OCR for {filename}: {policy_decision.reason}",
            )

    # Execute ocr_image using the callable from vision tools
    try:
        result = await _ocr_image_callable(ocr_provider, image_loader)(ocr_tool_call.arguments)
        return OCRResponse(**result)
    except Exception as exc:
        logger.exception("Error running OCR on %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed for {filename}: {exc}",
        )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    request: AnalyzeRequest = Depends(),
) -> AnalyzeResponse:
    """Run vision analysis on an uploaded image."""
    workspace = get_workspace()
    image_loader = get_image_loader(workspace)
    vision_provider = get_vision_provider()
    policy_engine = get_policy_engine()

    filename = file.filename or "uploaded_image.png"

    # Validate workspace path security
    try:
        resolved = workspace.resolve(filename)
    except WorkspaceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image filename: {exc}",
        )

    content = await file.read()

    # Check policy for image write
    tool_call_write = ToolCall(
        call_id=f"write_file_{filename}",
        tool_name="write_file",
        arguments={"path": filename},
    )
    if policy_engine is not None:
        policy_decision = policy_engine.evaluate(tool_call_write)
        if not policy_decision.is_allowed():
            logger.warning("Policy denied image write for %s: %s", filename, policy_decision.reason)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Policy denied upload for {filename}: {policy_decision.reason}",
            )

    # Save raw binary bytes to workspace
    try:
        resolved.write_bytes(content)
    except OSError as exc:
        logger.error("Error writing image file %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write image file {filename}",
        )

    # Create a tool call for analyze_image
    analyze_tool_call = ToolCall(
        call_id=f"analyze_image_{filename}",
        tool_name="analyze_image",
        arguments={
            "path": filename,
            "prompt": request.prompt or "",
        },
    )

    # Check policy for analyze_image
    if policy_engine is not None:
        policy_decision = policy_engine.evaluate(analyze_tool_call)
        if not policy_decision.is_allowed():
            logger.warning("Policy denied analyze_image for %s: %s", filename, policy_decision.reason)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Policy denied vision analysis for {filename}: {policy_decision.reason}",
            )

    # Execute analyze_image using the callable from vision tools
    try:
        result = await _analyze_image_callable(vision_provider, image_loader)(analyze_tool_call.arguments)
        return AnalyzeResponse(**result)
    except Exception as exc:
        logger.exception("Error running vision analysis on %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision analysis failed for {filename}: {exc}",
        )