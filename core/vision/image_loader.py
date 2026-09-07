"""Workspace-backed image loader (Phase 5C).

The :class:`ImageLoader` is the **single** path that the OCR/Vision
subsystem uses to read images from disk. It delegates ALL path validation
to the existing :class:`core.tools.workspace.Workspace` — no second
security implementation is added.

Properties:

* Rejects absolute paths, traversal, drive letters, UNC paths
  (handled by :class:`Workspace`).
* Rejects unsupported content types.
* Rejects empty files.
* Rejects files larger than :data:`_MAX_IMAGE_BYTES`.
* Returns a safe :class:`ImageInput` with metadata suitable for logging.
* Never logs the image bytes themselves.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

from core.tools.workspace import Workspace, WorkspaceError
from core.vision.types import (
    SUPPORTED_IMAGE_CONTENT_TYPES,
    ImageInput,
    ImageMetadata,
)


_logger = logging.getLogger("sovereign-ai.vision.loader")


#: Maximum image size (bytes) the loader will read. Mirrors the limit
#: in the file tools for consistency.
_MAX_IMAGE_BYTES: int = 50 * 1024 * 1024  # 50 MiB


class ImageLoader:
    """Load images from a :class:`Workspace` with security guarantees.

    The loader reuses the existing workspace security boundary. It does
    not perform any path validation of its own.

    Example::

        workspace = Workspace(root_path="/data/workspace")
        loader = ImageLoader(workspace)
        image = loader.load("scans/page1.png")
        # image.bytes contains the raw PNG bytes
        # image.metadata.content_type == "image/png"
    """

    __slots__ = ("_workspace",)

    def __init__(self, workspace: Workspace) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError(
                f"ImageLoader requires a Workspace, got {type(workspace).__name__}"
            )
        self._workspace = workspace

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def load(
        self,
        workspace_relative_path: str,
        *,
        content_type: Optional[str] = None,
    ) -> ImageInput:
        """Load an image from the workspace.

        Args:
            workspace_relative_path: Path relative to the workspace root.
                Must satisfy the workspace's path validation rules.
            content_type: Optional explicit content type. If not given,
                it is inferred from the file extension.

        Returns:
            An :class:`ImageInput` with bytes and safe metadata.

        Raises:
            WorkspaceError: if the path fails workspace validation.
            FileNotFoundError: if the file does not exist.
            UnsupportedImageError: if the content type is not supported.
            ValueError: if the image is empty or too large.
        """
        # Step 1: resolve through the workspace (single security gate)
        try:
            resolved = self._workspace.resolve(workspace_relative_path)
        except WorkspaceError as exc:
            _logger.info(
                "image_loader.path_rejected  reason=%s  category=%s",
                type(exc).__name__,
                getattr(exc, "relative_path", ""),
            )
            raise

        if not resolved.exists():
            from core.tools.types import FileNotFoundError as RAGFileNotFound

            raise RAGFileNotFound(
                f"File not found in workspace: {workspace_relative_path!r}",
                path=workspace_relative_path,
            )

        if not resolved.is_file():
            from core.tools.types import UnsupportedFileError

            raise UnsupportedFileError(
                f"Not a regular file: {workspace_relative_path!r}",
                path=workspace_relative_path,
            )

        # Step 2: determine content type
        if content_type is None:
            content_type = self._infer_content_type(resolved)
        if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
            from core.vision.errors import UnsupportedImageError

            raise UnsupportedImageError(
                f"Unsupported image content type: {content_type!r}",
            )

        # Step 3: read the file
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            from core.tools.types import FileToolError

            raise FileToolError(
                f"Cannot read file: {exc}",
                path=workspace_relative_path,
            ) from exc

        if not raw:
            from core.vision.errors import EmptyImageError

            raise EmptyImageError("Image file is empty")

        if len(raw) > _MAX_IMAGE_BYTES:
            from core.vision.errors import InvalidImageError

            raise InvalidImageError(
                f"Image is too large ({len(raw)} bytes; "
                f"max {_MAX_IMAGE_BYTES})"
            )

        # Step 4: build safe metadata
        image_id = self._stable_id(workspace_relative_path)
        metadata = ImageMetadata(
            image_id=image_id,
            filename=Path(workspace_relative_path).name,
            source_path=workspace_relative_path,
            content_type=content_type,
            width=None,
            height=None,
            size_bytes=len(raw),
        )

        _logger.info(
            "image_loader.loaded  image_id=%s  size_bytes=%d  content_type=%s",
            image_id,
            len(raw),
            content_type,
        )

        return ImageInput(metadata=metadata, bytes=raw)

    def load_from_bytes(
        self,
        data: bytes,
        *,
        source: str,
        content_type: str,
    ) -> ImageInput:
        """Build an :class:`ImageInput` from in-memory bytes.

        Use this when the image bytes are already in memory (e.g. the
        agent received them through some other channel). The ``source``
        argument is used only to build the image's metadata identifier
        and filename; no filesystem access occurs.

        Args:
            data: The image bytes.
            source: An application-meaningful source identifier (typically
                a workspace-relative path or a logical name).
            content_type: The MIME type of the image.

        Returns:
            An :class:`ImageInput` with the bytes and safe metadata.
        """
        if not data:
            from core.vision.errors import EmptyImageError

            raise EmptyImageError("Image bytes are empty")

        if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
            from core.vision.errors import UnsupportedImageError

            raise UnsupportedImageError(
                f"Unsupported image content type: {content_type!r}"
            )

        image_id = self._stable_id(source)
        metadata = ImageMetadata(
            image_id=image_id,
            filename=Path(source).name if source else "image",
            source_path=source,
            content_type=content_type,
            width=None,
            height=None,
            size_bytes=len(data),
        )
        return ImageInput(metadata=metadata, bytes=data)

    # ----------------------------------------------------------------- Internals

    @staticmethod
    def _stable_id(source: str) -> str:
        """Stable hash of a source identifier for use as an image_id."""
        joined = source or "unknown"
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _infer_content_type(path: Path) -> str:
        """Infer content type from file extension.

        Returns ``"application/octet-stream"`` if the extension is unknown.
        The caller validates that the result is in
        :data:`SUPPORTED_IMAGE_CONTENT_TYPES`.
        """
        ext = path.suffix.lower()
        mapping = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        return mapping.get(ext, "application/octet-stream")
