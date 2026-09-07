"""OCR providers for the Vision subsystem (Phase 5C).

An :class:`OCRProvider` extracts text from images. The interface is deliberately
abstract — a real provider would use a local model (Tesseract, EasyOCR,
PaddleOCR, or a vision-LLM via Ollama), while a fake provider generates
deterministic test output.

For this phase we ship a :class:`FakeOCRProvider` that produces deterministic
placeholder text suitable for tests and development on the Ryzen 3 laptop.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Final, Sequence

from core.vision.types import (
    ImageDocument,
    ImageInput,
    OCRBlock,
    OCRBoundingBox,
    OCRLine,
    OCRResult,
    OCRWord,
    _now_utc,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum image size (bytes) that the OCR subsystem will process.
_MAX_IMAGE_BYTES: Final[int] = 50 * 1024 * 1024  # 50 MiB

#: Minimum image size (bytes) that the OCR subsystem will process.
_MIN_IMAGE_BYTES: Final[int] = 1

#: Content types that OCR providers should recognise.
SUPPORTED_OCR_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class OCRProvider(ABC):
    """Abstract OCR provider.

    An OCR provider takes an image (or multi-page document) and returns
    extracted text. The interface is modality-agnostic: a future provider
    might use a local LLM for handwriting, a traditional OCR engine for
    print, or both in combination.
    """

    @abstractmethod
    def recognize(self, image: ImageInput) -> OCRResult:
        """Recognize text in a single image.

        Args:
            image: The image to process. Must have valid metadata and bytes.

        Returns:
            An :class:`OCRResult` with the extracted text and any
            available structured metadata (blocks, lines, boxes).

        Raises:
            UnsupportedImageError: if the image content type is not supported.
            InvalidImageError: if the image bytes are malformed.
            EmptyImageError: if the image has no bytes.
        """

    def recognize_document(
        self, document: ImageDocument
    ) -> list[OCRResult]:
        """Recognize text in a multi-page document.

        The default implementation calls :meth:`recognize` for each page.
        Subclasses may override this for batch-optimised processing.

        Args:
            document: The multi-page document to process.

        Returns:
            A list of :class:`OCRResult` objects, one per page, in order.
        """
        return [self.recognize(page) for page in document.pages]

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """A short machine-readable name for this provider (e.g. ``"fake"``)."""


# ---------------------------------------------------------------------------
# Fake / deterministic provider for tests
# ---------------------------------------------------------------------------


class FakeOCRProvider(OCRProvider):
    """A deterministic fake OCR provider for testing and development.

    The fake provider produces stable, deterministic output:

    * ``full_text`` is a function of the image's ``image_id``.
    * ``confidence`` is always ``0.95`` (high enough to be realistic).
    * ``blocks`` contains a single structured block.
    * The text is a deterministic function of the image bytes + a configurable
      seed, so identical images always produce identical output.

    This is NOT suitable for production. It exists solely to exercise the
    OCR pipeline without downloading a real OCR model.
    """

    __slots__ = ("_seed", "_dimension")

    def __init__(
        self,
        seed: int = 123,
    ) -> None:
        self._seed = seed

    @property
    def provider_name(self) -> str:
        return "fake"

    def recognize(self, image: ImageInput) -> OCRResult:
        # Validate image
        if not image.bytes:
            from core.vision.errors import EmptyImageError

            raise EmptyImageError("Image bytes are empty", image_id=image.image_id)

        if image.content_type not in SUPPORTED_OCR_CONTENT_TYPES:
            from core.vision.errors import UnsupportedImageError

            raise UnsupportedImageError(
                f"Unsupported content type: {image.content_type!r}",
                image_id=image.image_id,
            )

        # Derive deterministic text from image bytes
        text = self._derive_text(image.bytes, image.image_id)

        # Build a single structured block
        block = OCRBlock(
            text=text,
            confidence=0.95,
            lines=(
                OCRLine(
                    text=text,
                    confidence=0.95,
                    words=(
                        OCRWord(text=text, confidence=0.95),
                    ),
                ),
            ),
        )

        return OCRResult(
            image_id=image.image_id,
            full_text=text,
            confidence=0.95,
            blocks=(block,),
            provider=self.provider_name,
            recognized_at=_now_utc(),
        )

    def recognize_document(
        self, document: ImageDocument
    ) -> list[OCRResult]:
        if not document.pages:
            return []
        return [
            self.recognize(page)
            for page in document.pages
        ]

    # ----------------------------------------------------------------- Internals

    def _derive_text(self, data: bytes, image_id: str) -> str:
        """Derive deterministic placeholder text from image bytes and ID."""
        seed_bytes = hashlib.sha256(
            f"{self._seed}:{image_id}:{len(data)}".encode("utf-8")
        ).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "big")
        # Use a small pool of placeholder texts keyed to the seed
        placeholders = [
            "Extracted text from document.",
            "Recognised text block from image.",
            "Document content processed via OCR.",
            "Text extracted from scanned page.",
            "Page content recognised by OCR provider.",
        ]
        idx = seed_int % len(placeholders)
        return placeholders[idx]
