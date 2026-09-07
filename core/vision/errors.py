"""Error types for the OCR + Vision subsystem (Phase 5C).

All errors raised by the OCR/Vision layer are subclasses of
:class:`VisionError`. The exceptions are safe to surface to end users:
they never echo back raw image bytes, OCR text, or other sensitive
content. They may carry a stable ``image_id`` for diagnostics.
"""
from __future__ import annotations

from typing import Optional


class VisionError(Exception):
    """Base class for all OCR/Vision errors."""

    def __init__(
        self,
        message: str,
        image_id: Optional[str] = None,
    ) -> None:
        self.image_id = image_id
        super().__init__(message)


class UnsupportedImageError(VisionError):
    """The image content type is not supported by the provider."""


class InvalidImageError(VisionError):
    """The image bytes are malformed or unreadable."""


class EmptyImageError(VisionError):
    """The image has zero bytes."""


class ProviderUnavailableError(VisionError):
    """The provider could not be reached or is not configured."""


class VisionToolError(VisionError):
    """The vision tool was invoked with invalid arguments."""
