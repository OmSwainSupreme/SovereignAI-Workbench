"""Ollama-backed vision provider for the OCR + Vision subsystem (Phase 5C).

This provider implements the :class:`core.vision.vision.VisionProvider` interface
using a local Ollama server with a vision-capable model such as ``qwen2.5vl:3b``
or ``llava``. The server MUST run on the same machine (loopback) — the provider
refuses non-loopback base URLs at construction time, preserving the air-gapped
guarantee of the platform.

The provider talks only to ``/api/generate`` on the configured local Ollama
instance. It never contacts an external host. Image bytes are base64-encoded
into the request and are never logged.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, Optional

import httpx

from core.llm.errors import (
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from core.vision.errors import EmptyImageError, UnsupportedImageError
from core.vision.types import (
    ImageInput,
    VisionResult,
    _now_utc,
)
from core.vision.vision import SUPPORTED_VISION_CONTENT_TYPES, VisionProvider

_logger = logging.getLogger("sovereign-ai.vision.ollama")

# Hosts considered loopback (used for URL validation).
_LOOPBACK_RE = re.compile(
    r"^(?:127\.\d+\.\d+\.\d+|localhost|::1|\[::1\])$",
    re.IGNORECASE,
)


def _validate_loopback(url: str) -> str:
    """Validate that ``url`` points at a loopback host. Raise ConfigurationError otherwise."""
    try:
        parsed = httpx.URL(url)
        host = parsed.host or ""
        if not _LOOPBACK_RE.match(host):
            raise ConfigurationError(
                f"Ollama base_url must use a loopback host (127.0.0.1, ::1, localhost). "
                f"Got: {url!r}  This is a security requirement to prevent accidental "
                f"data exfiltration to external hosts."
            )
        return url.rstrip("/")
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid Ollama base_url: {url!r}") from exc


class OllamaVisionProvider(VisionProvider):
    """Vision provider backed by a local Ollama server.

    Uses a vision-capable model (e.g. ``qwen2.5vl:3b``) to analyse an image and
    return a natural-language description prompted by the caller. The prompt is
    sent verbatim when supplied; a generic description prompt is used otherwise.

    The provider is **synchronous** — ``analyze`` blocks until the local Ollama
    server responds. This matches the :class:`VisionProvider` interface (the
    agent tool bridge calls these methods without ``await``).

    The constructor accepts the same keyword arguments as the Ollama provider's
    configuration so it can be configured consistently with ``core/llm``.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "",
        request_timeout_seconds: int = 120,
    ) -> None:
        self._base_url = _validate_loopback(base_url)
        self._default_model = default_model
        self._timeout_seconds = max(1, request_timeout_seconds)
        self._client: httpx.Client = httpx.Client(
            timeout=httpx.Timeout(self._timeout_seconds),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    @property
    def provider_name(self) -> str:
        """A short machine-readable name for this provider."""
        return "ollama_vision"

    @property
    def default_model(self) -> Optional[str]:
        """The configured default vision model, or ``None``."""
        return self._default_model or None

    def close(self) -> None:
        """Close the underlying HTTP client. Call at application shutdown."""
        self._client.close()

    def analyze(
        self,
        image: ImageInput,
        prompt: str = "",
    ) -> VisionResult:
        """Analyse the given image using Ollama's vision capabilities.

        Args:
            image: The image to analyse. Must have valid metadata and bytes.
            prompt: Optional natural-language prompt guiding the analysis
                (e.g. ``"Describe this image"``). May be empty for a generic
                description.

        Returns:
            A :class:`VisionResult` with the model's description.

        Raises:
            EmptyImageError: if the image has no bytes.
            UnsupportedImageError: if the image content type is not supported.
            ConfigurationError: if no default model is configured.
            ProviderError: for network / Ollama failures.
        """
        self._validate_image(image)

        model = self._default_model
        if not model:
            raise ConfigurationError(
                "No model specified and no default model is configured"
            )

        instruction = prompt.strip() if prompt else "Describe this image in detail."
        payload: dict[str, Any] = {
            "model": model,
            "prompt": instruction,
            "images": [base64.b64encode(image.bytes).decode("utf-8")],
            "stream": False,
        }

        start = time.monotonic()
        try:
            data = self._post("/api/generate", payload)
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            _logger.info(
                "ollama_vision.analyze  model=%s  latency_ms=%.1f",
                model,
                latency_ms,
            )

        description = data.get("response", "").strip()

        # A real vision-LLM may describe regions; we surface the raw text as the
        # description and leave structured regions/tags empty — the caller can
        # re-prompt for specifics. This mirrors the extensible VisionResult shape.
        return VisionResult(
            image_id=image.image_id,
            prompt=prompt,
            description=description,
            regions=(),
            tags=(),
            confidence=None,
            provider=self.provider_name,
            analyzed_at=_now_utc(),
        )

    # ----------------------------------------------------------------- Helpers

    @staticmethod
    def _validate_image(image: ImageInput) -> None:
        if not image.bytes:
            raise EmptyImageError("Image bytes are empty", image_id=image.image_id)
        if image.content_type not in SUPPORTED_VISION_CONTENT_TYPES:
            raise UnsupportedImageError(
                f"Unsupported content type: {image.content_type!r}",
                image_id=image.image_id,
            )

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """POST to ``path`` and return the parsed JSON body.

        Raises:
            ProviderUnavailableError: on connection failure.
            ProviderTimeoutError: on timeout.
            ProviderResponseError: on non-2xx or JSON decode error.
            ProviderError: on unexpected HTTP errors.
        """
        _logger.debug("POST %s  payload_keys=%s", path, list(payload.keys()))
        try:
            response = self._client.post(
                self._url(path),
                json=payload,
                timeout=timeout,
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ProviderTimeoutError(
                f"Ollama request to {path} timed out after "
                f"{self._timeout_seconds}s"
            ) from exc
        except httpx.ConnectTimeout as exc:
            raise ProviderUnavailableError(
                f"Connection to Ollama timed out: {exc}"
            ) from exc
        except httpx.PoolTimeout as exc:
            raise ProviderError(
                f"Ollama request failed (pool exhausted): {exc}"
            ) from exc
        except httpx.WriteTimeout as exc:
            raise ProviderError(
                f"Ollama request failed (write timeout): {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        if response.status_code == 404:
            model = payload.get("model", "(unknown)")
            raise ModelNotFoundError(model)
        if response.status_code >= 500:
            raise ProviderError(
                f"Ollama server error {response.status_code}: {response.text[:200]}"
            )
        if not response.is_success:
            raise ProviderResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"Ollama returned non-JSON response: {exc}. Body: {response.text[:200]}"
            ) from exc