"""OCR and Vision tools for the agent tool registry (Phase 5C).

This module provides two :class:`ToolDefinition` objects plus their
callable executors, wired into the :class:`DefaultToolRegistry` via
:func:`register_vision_tools`.

The tools follow the same pattern as :mod:`core.rag.tools`:

* :data:`OCR_IMAGE_TOOL` — runs OCR on an image at a workspace-relative path.
* :data:`ANALYZE_IMAGE_TOOL` — runs vision analysis on an image.
* :func:`register_vision_tools` — wires both into a registry.

The tools:

* Reject paths outside the workspace (delegated to :class:`ImageLoader`).
* Reject unsupported content types.
* Return structured results with no raw image bytes.
* Do not log OCR text, vision descriptions, or image content.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core.agent.errors import ToolExecutionError
from core.agent.interfaces import ToolDefinition
from core.agent.registry import DefaultToolRegistry
from core.agent.types import ToolCall, ToolResult
from core.vision.image_loader import ImageLoader
from core.vision.ocr import OCRProvider
from core.vision.types import ImageInput
from core.vision.vision import VisionProvider


_logger = logging.getLogger("sovereign-ai.vision.tools")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


OCR_IMAGE_TOOL = ToolDefinition(
    name="ocr_image",
    description=(
        "Run OCR (text recognition) on an image inside the workspace. "
        "Returns the extracted text and any structured blocks/lines the "
        "provider produced. The path must be workspace-relative."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative path to the image to OCR. "
                    "Must point at a supported image file."
                ),
            },
            "language": {
                "type": "string",
                "description": (
                    "Optional language hint (e.g. 'en'). "
                    "Providers that don't support language hints ignore it."
                ),
            },
        },
        "required": ["path"],
    },
    output_description=(
        "A structured OCR result containing the full text, confidence, "
        "and any blocks/lines the provider produced."
    ),
    capability="vision",
)


ANALYZE_IMAGE_TOOL = ToolDefinition(
    name="analyze_image",
    description=(
        "Analyse the visual content of an image inside the workspace. "
        "Returns a textual description, optional regions of interest, "
        "and optional tags. The path must be workspace-relative."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative path to the image to analyse. "
                    "Must point at a supported image file."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Optional natural-language instruction (e.g. "
                    "'Describe the people in this image')."
                ),
            },
        },
        "required": ["path"],
    },
    output_description=(
        "A structured vision result containing a description, confidence, "
        "regions, and tags."
    ),
    capability="vision",
)


# ---------------------------------------------------------------------------
# Callable bridges
# ---------------------------------------------------------------------------


def _ocr_image_callable(
    ocr_provider: OCRProvider,
    image_loader: ImageLoader,
):
    """Return an async callable that runs OCR via the supplied provider."""

    async def _call(args: Mapping[str, Any]) -> object:
        path = args.get("path", "")
        language = args.get("language", "")

        if not path or not isinstance(path, str):
            raise ToolExecutionError(
                "ocr_image requires a 'path' argument"
            )

        # Load via the workspace; raises WorkspaceError for bad paths.
        image = image_loader.load(path)

        # Run OCR.
        result = ocr_provider.recognize(image)

        # Convert to a plain serialisable dict for the agent.
        return {
            "image_id": result.image_id,
            "full_text": result.full_text,
            "confidence": result.confidence,
            "page_number": result.page_number,
            "page_count": result.page_count,
            "language": result.language or language or None,
            "block_count": len(result.blocks),
            "blocks": [
                {
                    "text": b.text,
                    "confidence": b.confidence,
                    "block_type": b.block_type,
                }
                for b in result.blocks
            ],
            "provider": result.provider,
        }

    return _call


def _analyze_image_callable(
    vision_provider: VisionProvider,
    image_loader: ImageLoader,
):
    """Return an async callable that runs vision analysis."""

    async def _call(args: Mapping[str, Any]) -> object:
        path = args.get("path", "")
        prompt = args.get("prompt", "")

        if not path or not isinstance(path, str):
            raise ToolExecutionError(
                "analyze_image requires a 'path' argument"
            )

        # Load via the workspace.
        image = image_loader.load(path)

        # Run analysis.
        result = vision_provider.analyze(image, prompt)

        return {
            "image_id": result.image_id,
            "description": result.description,
            "confidence": result.confidence,
            "tags": list(result.tags),
            "regions": [
                {
                    "label": r.label,
                    "confidence": r.confidence,
                }
                for r in result.regions
            ],
            "provider": result.provider,
        }

    return _call


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def register_vision_tools(
    registry: DefaultToolRegistry,
    ocr_provider: OCRProvider,
    vision_provider: VisionProvider,
    image_loader: ImageLoader,
) -> None:
    """Register the OCR and vision tools into ``registry``.

    After this call, the agent's tool registry will include both tools,
    wired to the supplied providers and image loader.

    Args:
        registry: A :class:`DefaultToolRegistry` to populate.
        ocr_provider: The OCR provider to use.
        vision_provider: The vision provider to use.
        image_loader: The image loader to use (binds the workspace).

    Raises:
        TypeError: if ``registry`` is not a :class:`DefaultToolRegistry`.
    """
    if not isinstance(registry, DefaultToolRegistry):
        raise TypeError(
            "register_vision_tools requires a DefaultToolRegistry "
            "(other registries do not support callables)"
        )

    registry.register(OCR_IMAGE_TOOL, _ocr_image_callable(ocr_provider, image_loader))
    registry.register(
        ANALYZE_IMAGE_TOOL, _analyze_image_callable(vision_provider, image_loader)
    )

    _logger.info(
        "vision.tools.registered  ocr=%s  vision=%s",
        ocr_provider.provider_name,
        vision_provider.provider_name,
    )
