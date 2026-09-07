"""Tests for the Ollama vision and OCR providers (Phase 5C) + routing integration.

These tests use a fake httpx transport to simulate the Ollama server's
``/api/generate`` multimodal behaviour without requiring a real server or
a real vision model.

Coverage includes:
* provider construction, defaults, and loopback-only security
* successful analysis / recognition
* input validation (empty bytes, unsupported content types)
* missing default model
* provider-failure mapping (connect, timeout, HTTP errors, malformed JSON)
* end-to-end ``image -> router -> provider -> result`` integration
"""
from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from core.llm import (
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from core.routing import Capability, Modality, ModelDefinition, ModelRouter
from core.routing.registry import ModelRegistry
from core.vision.errors import EmptyImageError, UnsupportedImageError
from core.vision.ollama_ocr import OllamaOCRProvider
from core.vision.ollama_vision import OllamaVisionProvider
from core.vision.types import (
    ImageInput,
    ImageMetadata,
    OCRResult,
    VisionResult,
    build_vision_routing_request,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_FAKE_IMAGE_BYTES = _PNG_MAGIC + b"x" * 128


def _make_image(
    image_id: str = "img-test",
    source: str = "scans/page.png",
    content_type: str = "image/png",
    data: bytes = _FAKE_IMAGE_BYTES,
) -> ImageInput:
    """Build an :class:`ImageInput` for tests."""
    return ImageInput(
        metadata=ImageMetadata(
            image_id=image_id,
            filename=source.split("/")[-1],
            source_path=source,
            content_type=content_type,
            size_bytes=len(data),
        ),
        bytes=data,
    )


def make_response(
    status_code: int = 200,
    json_data: Any = None,
    text: str = "",
) -> httpx.Response:
    """Construct an httpx.Response for use in a fake transport."""
    if json_data is not None:
        return httpx.Response(status_code=status_code, json=json_data)
    return httpx.Response(
        status_code=status_code,
        text=text or json.dumps({"error": "no body"}),
    )


def make_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    """Build a MockTransport that delegates to ``handler(request)``."""
    return httpx.MockTransport(handler)


def _bind(provider, handler) -> None:
    """Replace the provider's HTTP client with one using a fake transport."""
    provider._client = httpx.Client(  # type: ignore[assignment]
        transport=make_transport(handler),
        base_url="http://127.0.0.1:11434",
    )


def _make_vision_router() -> ModelRouter:
    """Build a :class:`ModelRouter` with a vision-capable model.

    Mirrors ``config/models.yaml`` — the vision model is ``qwen2.5vl:3b``
    with the VISION capability and TEXT+IMAGE input modalities.
    """
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            logical_name="vision",
            provider="ollama",
            provider_model="qwen2.5vl:3b",
            capabilities=frozenset({Capability.VISION, Capability.GENERAL}),
            input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
    )
    return ModelRouter(registry)


# ---------------------------------------------------------------------------
# Vision provider
# ---------------------------------------------------------------------------


class TestOllamaVisionProvider:
    def test_constructs_with_defaults(self) -> None:
        provider = OllamaVisionProvider()
        assert provider.provider_name == "ollama_vision"
        assert provider.default_model is None

    def test_constructs_with_kwargs(self) -> None:
        provider = OllamaVisionProvider(
            base_url="http://localhost:11434",
            default_model="qwen2.5vl:3b",
            request_timeout_seconds=60,
        )
        assert provider.default_model == "qwen2.5vl:3b"

    def test_refuses_non_loopback_base_url(self) -> None:
        with pytest.raises(ConfigurationError, match="loopback"):
            OllamaVisionProvider(base_url="http://10.0.0.5:11434")

    def test_analyze_returns_vision_result(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return make_response(
                200,
                json_data={
                    "model": "qwen2.5vl:3b",
                    "response": "A red apple on a wooden table.",
                    "done": True,
                },
            )

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            result = provider.analyze(_make_image(), "describe")
        finally:
            provider.close()

        assert isinstance(result, VisionResult)
        assert result.image_id == "img-test"
        assert result.prompt == "describe"
        assert result.description == "A red apple on a wooden table."
        assert result.provider == "ollama_vision"
        # Confirm the request hit /api/generate and carried the image as base64.
        assert len(captured) == 1
        assert captured[0].url.path == "/api/generate"
        body = json.loads(captured[0].content)
        assert body["model"] == "qwen2.5vl:3b"
        assert body["images"] and len(body["images"]) == 1

    def test_analyze_default_prompt_when_empty(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return make_response(200, json_data={"response": "ok", "done": True})

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            provider.analyze(_make_image())
        finally:
            provider.close()

        assert captured[0]["prompt"] != ""
        assert "describe" in captured[0]["prompt"].lower()

    def test_analyze_rejects_empty_bytes(self) -> None:
        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        with pytest.raises(EmptyImageError):
            provider.analyze(_make_image(data=b""))

    def test_analyze_rejects_unsupported_content_type(self) -> None:
        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        with pytest.raises(UnsupportedImageError):
            provider.analyze(_make_image(content_type="image/gif"))

    def test_analyze_requires_default_model(self) -> None:
        provider = OllamaVisionProvider()
        _bind(
            provider,
            lambda req: make_response(200, json_data={"response": "x", "done": True}),
        )
        with pytest.raises(ConfigurationError, match="No model"):
            provider.analyze(_make_image())


# ---------------------------------------------------------------------------
# Vision provider error mapping
# ---------------------------------------------------------------------------


class TestOllamaVisionProviderErrors:
    def test_connect_error_maps_to_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderUnavailableError):
                provider.analyze(_make_image(), "describe")
        finally:
            provider.close()

    def test_read_timeout_maps_to_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out")

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderTimeoutError):
                provider.analyze(_make_image(), "describe")
        finally:
            provider.close()

    def test_connect_timeout_maps_to_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("Connect timed out")

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderUnavailableError):
                provider.analyze(_make_image(), "describe")
        finally:
            provider.close()

    def test_404_maps_to_model_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(404, text="model not found")

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                provider.analyze(_make_image(), "describe")
            assert exc_info.value.model == "qwen2.5vl:3b"
        finally:
            provider.close()

    def test_500_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(500, text="internal error")

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderError):
                provider.analyze(_make_image(), "describe")
        finally:
            provider.close()

    def test_malformed_json_maps_to_response_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(200, text="not json {")

        provider = OllamaVisionProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderResponseError):
                provider.analyze(_make_image(), "describe")
        finally:
            provider.close()


# ---------------------------------------------------------------------------
# OCR provider
# ---------------------------------------------------------------------------


class TestOllamaOCRProvider:
    def test_constructs_with_defaults(self) -> None:
        provider = OllamaOCRProvider()
        assert provider.provider_name == "ollama_ocr"
        assert provider.default_model is None

    def test_constructs_with_kwargs(self) -> None:
        provider = OllamaOCRProvider(
            base_url="http://localhost:11434",
            default_model="qwen2.5vl:3b",
            request_timeout_seconds=60,
        )
        assert provider.default_model == "qwen2.5vl:3b"

    def test_refuses_non_loopback_base_url(self) -> None:
        with pytest.raises(ConfigurationError, match="loopback"):
            OllamaOCRProvider(base_url="http://10.0.0.5:11434")

    def test_recognize_returns_ocr_result(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return make_response(
                200,
                json_data={
                    "model": "qwen2.5vl:3b",
                    "response": "Invoice #1234 \nTotal: $50.00",
                    "done": True,
                },
            )

        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            result = provider.recognize(_make_image())
        finally:
            provider.close()

        assert isinstance(result, OCRResult)
        assert result.image_id == "img-test"
        assert "Invoice" in result.full_text
        assert result.provider == "ollama_ocr"
        # Confirm the request hit /api/generate and carried the image.
        assert len(captured) == 1
        assert captured[0].url.path == "/api/generate"
        body = json.loads(captured[0].content)
        assert body["model"] == "qwen2.5vl:3b"
        assert body["images"] and len(body["images"]) == 1

    def test_recognize_rejects_empty_bytes(self) -> None:
        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        with pytest.raises(EmptyImageError):
            provider.recognize(_make_image(data=b""))

    def test_recognize_rejects_unsupported_content_type(self) -> None:
        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        with pytest.raises(UnsupportedImageError):
            provider.recognize(_make_image(content_type="image/gif"))

    def test_recognize_requires_default_model(self) -> None:
        provider = OllamaOCRProvider()
        _bind(
            provider,
            lambda req: make_response(200, json_data={"response": "x", "done": True}),
        )
        with pytest.raises(ConfigurationError, match="No model"):
            provider.recognize(_make_image())


# ---------------------------------------------------------------------------
# OCR provider error mapping
# ---------------------------------------------------------------------------


class TestOllamaOCRProviderErrors:
    def test_connect_error_maps_to_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderUnavailableError):
                provider.recognize(_make_image())
        finally:
            provider.close()

    def test_read_timeout_maps_to_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out")

        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderTimeoutError):
                provider.recognize(_make_image())
        finally:
            provider.close()

    def test_404_maps_to_model_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(404, text="model not found")

        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                provider.recognize(_make_image())
            assert exc_info.value.model == "qwen2.5vl:3b"
        finally:
            provider.close()

    def test_500_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(500, text="internal error")

        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderError):
                provider.recognize(_make_image())
        finally:
            provider.close()

    def test_malformed_json_maps_to_response_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(200, text="not json {")

        provider = OllamaOCRProvider(default_model="qwen2.5vl:3b")
        _bind(provider, handler)
        try:
            with pytest.raises(ProviderResponseError):
                provider.recognize(_make_image())
        finally:
            provider.close()


# ---------------------------------------------------------------------------
# End-to-end routing integration: image -> router -> provider -> result
# ---------------------------------------------------------------------------


class TestRoutingIntegration:
    """Verifies the vision/OCR pipeline through the router.

    The router selects the vision-capable model from the registry (mirroring
    ``config/models.yaml``); the provider is instantiated with the model the
    router picked and produces the analysis / OCR result.
    """

    def test_router_selects_vision_model(self) -> None:
        router = _make_vision_router()
        decision = router.route(build_vision_routing_request())
        assert decision.model.logical_name == "vision"
        assert decision.model.provider == "ollama"
        assert decision.model.provider_model == "qwen2.5vl:3b"
        assert decision.modality_satisfied

    def test_vision_end_to_end_image_to_result(self) -> None:
        """A vision task routes to the vision model, then analyse() works."""
        router = _make_vision_router()
        provider_model = router.route(build_vision_routing_request()).model.provider_model
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return make_response(
                200,
                json_data={
                    "model": provider_model,
                    "response": "A test image description.",
                    "done": True,
                },
            )

        provider = OllamaVisionProvider(default_model=provider_model)
        _bind(provider, handler)
        try:
            result = provider.analyze(_make_image(), "describe")
        finally:
            provider.close()

        assert isinstance(result, VisionResult)
        assert result.description == "A test image description."
        assert result.provider == "ollama_vision"
        # The provider hit /api/generate with the router-selected model.
        assert len(captured) == 1
        assert captured[0].url.path == "/api/generate"
        body = json.loads(captured[0].content)
        assert body["model"] == "qwen2.5vl:3b"

    def test_ocr_end_to_end_image_to_result(self) -> None:
        """An OCR task routes to the vision model, then recognize() works."""
        router = _make_vision_router()
        provider_model = router.route(build_vision_routing_request()).model.provider_model
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return make_response(
                200,
                json_data={
                    "model": provider_model,
                    "response": "Extracted text from image.",
                    "done": True,
                },
            )

        provider = OllamaOCRProvider(default_model=provider_model)
        _bind(provider, handler)
        try:
            result = provider.recognize(_make_image())
        finally:
            provider.close()

        assert isinstance(result, OCRResult)
        assert result.full_text == "Extracted text from image."
        assert result.provider == "ollama_ocr"
        # The provider hit /api/generate with the router-selected model.
        assert len(captured) == 1
        assert captured[0].url.path == "/api/generate"
        body = json.loads(captured[0].content)
        assert body["model"] == "qwen2.5vl:3b"