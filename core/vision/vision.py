"""Vision providers for the Vision subsystem (Phase 5C).

A :class:`VisionProvider` analyses image content and returns structured
results (description, regions, tags, etc.). The interface is deliberately
abstract — a real provider would use a local vision-LLM (LLaVA, Qwen-VL,
InternVL) via Ollama, or a local ONNX model.

For this phase we ship a :class:`FakeVisionProvider` that produces
deterministic placeholder output suitable for tests and development on
the Ryzen 3 laptop.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from core.vision.types import (
    ImageInput,
    OCRBoundingBox,
    VisionRegion,
    VisionResult,
    _now_utc,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum image size (bytes) the vision subsystem will process.
_MAX_IMAGE_BYTES: int = 50 * 1024 * 1024  # 50 MiB

#: Content types that vision providers should recognise.
SUPPORTED_VISION_CONTENT_TYPES: frozenset[str] = frozenset(
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


class VisionProvider(ABC):
    """Abstract vision provider.

    A vision provider takes an image and a prompt/instruction and
    returns a structured analysis (description, regions, tags, etc.).
    """

    @abstractmethod
    def analyze(
        self,
        image: ImageInput,
        prompt: str = "",
    ) -> VisionResult:
        """Analyse the given image.

        Args:
            image: The image to analyse. Must have valid metadata and bytes.
            prompt: Optional natural-language prompt guiding the analysis
                (e.g. ``"Describe this image"``). May be empty for a
                provider that always returns a generic description.

        Returns:
            A :class:`VisionResult` with the analysis.

        Raises:
            UnsupportedImageError: if the image content type is not supported.
            InvalidImageError: if the image bytes are malformed.
            EmptyImageError: if the image has no bytes.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """A short machine-readable name for this provider (e.g. ``"fake"``)."""


# ---------------------------------------------------------------------------
# Fake / deterministic provider for tests
# ---------------------------------------------------------------------------


class FakeVisionProvider(VisionProvider):
    """A deterministic fake vision provider for testing and development.

    The fake provider produces stable, deterministic output:

    * ``description`` is a function of the image's ``image_id`` and prompt.
    * ``regions`` contains a single region with a generic label.
    * ``tags`` is a small set of placeholder tags.
    * The same image + prompt always produces the same result.

    This is NOT suitable for production. It exists solely to exercise the
    vision pipeline without downloading a real vision model.
    """

    __slots__ = ("_seed",)

    def __init__(self, seed: int = 456) -> None:
        self._seed = seed

    @property
    def provider_name(self) -> str:
        return "fake"

    def analyze(
        self,
        image: ImageInput,
        prompt: str = "",
    ) -> VisionResult:
        if not image.bytes:
            from core.vision.errors import EmptyImageError

            raise EmptyImageError("Image bytes are empty", image_id=image.image_id)

        if image.content_type not in SUPPORTED_VISION_CONTENT_TYPES:
            from core.vision.errors import UnsupportedImageError

            raise UnsupportedImageError(
                f"Unsupported content type: {image.content_type!r}",
                image_id=image.image_id,
            )

        description = self._derive_description(image.bytes, image.image_id, prompt)
        tags = self._derive_tags(image.bytes, image.image_id)
        region = self._derive_region(image.bytes, image.image_id)

        return VisionResult(
            image_id=image.image_id,
            prompt=prompt,
            description=description,
            regions=(region,),
            tags=tags,
            confidence=0.90,
            provider=self.provider_name,
            analyzed_at=_now_utc(),
        )

    # ----------------------------------------------------------------- Internals

    def _derive_description(
        self, data: bytes, image_id: str, prompt: str
    ) -> str:
        """Derive a deterministic description from the image and prompt."""
        seed_bytes = hashlib.sha256(
            f"{self._seed}:{image_id}:{prompt}".encode("utf-8")
        ).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "big")
        descriptions = [
            "An image containing visual content of interest.",
            "A photograph with identifiable elements.",
            "Visual content suitable for further analysis.",
            "An image showing various visual features.",
        ]
        idx = seed_int % len(descriptions)
        if prompt:
            return f"{descriptions[idx]} (prompt: {prompt!r})"
        return descriptions[idx]

    def _derive_tags(self, data: bytes, image_id: str) -> tuple[str, ...]:
        """Derive a small set of deterministic tags."""
        seed_bytes = hashlib.sha256(
            f"{self._seed}:tags:{image_id}".encode("utf-8")
        ).digest()
        candidates = ["document", "photo", "outdoor", "indoor", "people", "text"]
        n = (seed_bytes[0] % 3) + 1
        return tuple(candidates[i] for i in range(n) if i < len(candidates))

    def _derive_region(self, data: bytes, image_id: str) -> VisionRegion:
        """Derive a single placeholder region."""
        seed_bytes = hashlib.sha256(
            f"{self._seed}:region:{image_id}".encode("utf-8")
        ).digest()
        labels = ["subject", "object", "region", "area"]
        idx = seed_bytes[1] % len(labels)
        return VisionRegion(
            label=labels[idx],
            confidence=0.85,
            box=OCRBoundingBox(x=0, y=0, width=100, height=100),
        )
