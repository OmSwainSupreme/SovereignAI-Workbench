"""Framework-agnostic data types for the OCR + Vision subsystem (Phase 5C).

These types are deliberately plain dataclasses. They must not depend on
FastAPI, Pydantic, numpy, PIL, or any other framework. The application
layer (``backend/app``) may wrap them in Pydantic DTOs for HTTP transport,
but this module remains pure.

This mirrors the design convention of :mod:`core.rag.types`,
:mod:`core.agent.types`, and :mod:`core.tools.types`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from core.routing.types import (
    Capability,
    Modality,
    RoutingRequest,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current UTC time, timezone-aware."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Image input types
# ---------------------------------------------------------------------------

#: Supported image content types. The system is intentionally narrow at this
#: stage; PDF rendering and additional formats will be added later.
SUPPORTED_IMAGE_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }
)

#: Mapping of content type -> conventional file extension. Used only for
#: validation; the system never infers content type from extension.
_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


@dataclass(frozen=True)
class ImageMetadata:
    """Safe, non-sensitive metadata about an image.

    The metadata deliberately omits image bytes. Dimensions and content
    type are safe to log; the actual image content is not.
    """

    image_id: str
    """Stable identifier for the image (e.g. derived from source path)."""

    filename: str
    """The image filename (no path)."""

    source_path: str
    """A workspace-relative source path or other application-meaningful
    identifier. Never an absolute host path."""

    content_type: str
    """MIME content type, e.g. ``"image/png"``."""

    width: Optional[int] = None
    """Image width in pixels, if known."""

    height: Optional[int] = None
    """Image height in pixels, if known."""

    size_bytes: int = 0
    """Size of the image in bytes, if known."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Additional safe metadata (e.g. page number for a multi-page document)."""


@dataclass(frozen=True)
class ImageInput:
    """An image to be processed by an OCR or vision provider.

    The :class:`ImageInput` carries the raw bytes plus safe metadata. The
    bytes are kept in a tuple of ``int`` to avoid mutation issues; in
    practice this is constructed once by the image loader and not copied
    around carelessly.

    Providers MUST NOT log the ``bytes`` field or any derived content
    (OCR text, vision descriptions, etc.).
    """

    metadata: ImageMetadata
    bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.bytes, (bytes, bytearray, memoryview)):
            raise TypeError("ImageInput.bytes must be a bytes-like object")
        if not self.metadata.content_type:
            raise ValueError("ImageInput.metadata.content_type is required")
        if not self.metadata.image_id:
            raise ValueError("ImageInput.metadata.image_id is required")

    @property
    def image_id(self) -> str:
        """Convenience: the image_id from the metadata."""
        return self.metadata.image_id

    @property
    def content_type(self) -> str:
        """Convenience: the content_type from the metadata."""
        return self.metadata.content_type


@dataclass(frozen=True)
class ImageDocument:
    """A multi-image document, e.g. a scanned multi-page document.

    Each page is represented as an :class:`ImageInput`. A document may
    have one or more pages. The document itself has a stable
    ``document_id`` (analogous to a RAG document) so that
    ``KnowledgeBase.ingest`` can later be used to ingest OCR output.
    """

    document_id: str
    """Stable identifier for the document (e.g. derived from source path)."""

    filename: str
    """The document's filename (no path)."""

    source_path: str
    """Workspace-relative source path or other application identifier."""

    pages: tuple[ImageInput, ...] = field(default_factory=tuple)
    """The document's pages, in order. Empty tuple for an empty document."""


# ---------------------------------------------------------------------------
# OCR result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRBoundingBox:
    """A rectangle in image coordinates (pixels)."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class OCRWord:
    """A single word detected by OCR, with optional confidence and box."""

    text: str
    confidence: Optional[float] = None
    """Confidence score in ``[0.0, 1.0]``. ``None`` if the provider did
    not produce a per-word score."""

    box: Optional[OCRBoundingBox] = None


@dataclass(frozen=True)
class OCRLine:
    """A line of text detected by OCR."""

    text: str
    confidence: Optional[float] = None
    words: tuple[OCRWord, ...] = field(default_factory=tuple)
    box: Optional[OCRBoundingBox] = None


@dataclass(frozen=True)
class OCRBlock:
    """A block of text detected by OCR (e.g. a paragraph, a header)."""

    text: str
    confidence: Optional[float] = None
    lines: tuple[OCRLine, ...] = field(default_factory=tuple)
    box: Optional[OCRBoundingBox] = None
    block_type: str = "text"
    """Free-form block kind: ``"text"``, ``"header"``, ``"table"``, etc."""


@dataclass(frozen=True)
class OCRResult:
    """The result of an OCR operation on a single image or multi-page document.

    OCR results preserve the full extracted text plus structured metadata
    (blocks, lines, words, bounding boxes) when the provider supplies it.
    Real providers may set confidence values; fake providers may omit them.
    """

    image_id: str
    """The :class:`ImageInput.image_id` this OCR result corresponds to."""

    full_text: str
    """The full extracted text, with block/line separators preserved
    (typically ``"\\n\\n"`` between blocks, ``"\\n"`` between lines)."""

    page_number: Optional[int] = None
    """1-based page number, or ``None`` for a single-page image."""

    page_count: Optional[int] = None
    """Total number of pages in the source document, or ``None`` if
    the provider does not know."""

    language: Optional[str] = None
    """Detected or declared language code (e.g. ``"en"``), if available."""

    confidence: Optional[float] = None
    """Aggregate confidence in ``[0.0, 1.0]``, if the provider reports one."""

    blocks: tuple[OCRBlock, ...] = field(default_factory=tuple)
    """Structured text blocks, if the provider produces them."""

    recognized_at: datetime = field(default_factory=_now_utc)
    """When the OCR was performed (UTC)."""

    provider: str = ""
    """Name of the provider that produced this result (e.g. ``"fake"``)."""


# ---------------------------------------------------------------------------
# Vision result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionRegion:
    """A detected region of interest in an image, with a label and score."""

    label: str
    """What the region is, e.g. ``"person"``, ``"table"``."""

    confidence: Optional[float] = None
    """Confidence score in ``[0.0, 1.0]``, if the provider reports one."""

    box: Optional[OCRBoundingBox] = None
    """Bounding box in image coordinates."""


@dataclass(frozen=True)
class VisionResult:
    """The result of a vision analysis operation.

    The result is intentionally extensible: providers may populate
    ``description``, ``regions``, or both, depending on what the model
    actually returns.
    """

    image_id: str
    """The :class:`ImageInput.image_id` this vision result corresponds to."""

    prompt: str
    """The prompt/instruction that was analysed against."""

    description: str = ""
    """A textual description or analysis of the image. May be empty if
    the provider only returns regions."""

    regions: tuple[VisionRegion, ...] = field(default_factory=tuple)
    """Detected regions, if the provider produces them."""

    confidence: Optional[float] = None
    """Aggregate confidence, if the provider reports one."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """Free-form tags or categories the provider assigned (e.g. ``"indoor"``)."""

    analyzed_at: datetime = field(default_factory=_now_utc)
    """When the analysis was performed (UTC)."""

    provider: str = ""
    """Name of the provider that produced this result (e.g. ``"fake"``)."""


# ---------------------------------------------------------------------------
# Routing integration helper
# ---------------------------------------------------------------------------


def build_vision_routing_request(
    *,
    preferred_model: Optional[str] = None,
    excluded_models: Optional[Sequence[str]] = None,
) -> RoutingRequest:
    """Build a :class:`RoutingRequest` for a vision task.

    The request asks for ``TaskType.VISION`` (which the existing router
    maps to the ``VISION`` capability and ``{TEXT, IMAGE}`` modalities).
    No new model-selection system is added — this just packages the
    standard request in a single helper.

    Args:
        preferred_model: Optional logical model name. Honoured by the
            router if it satisfies the VISION capability.
        excluded_models: Optional iterable of logical names to skip.

    Returns:
        A :class:`RoutingRequest` ready to be passed to
        :meth:`core.routing.router.ModelRouter.route`.
    """
    return RoutingRequest(
        task_type=TaskType.VISION,
        required_capabilities=frozenset({Capability.VISION}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        preferred_model=preferred_model,
        excluded_models=frozenset(excluded_models) if excluded_models else frozenset(),
    )
