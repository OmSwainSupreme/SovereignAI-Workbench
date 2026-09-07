"""Tests for the Ollama provider.

These tests use a fake httpx transport to simulate the Ollama server's
behaviour without requiring a real server.
"""
from __future__ import annotations

import json
from typing import Any

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
from core.llm.providers.ollama import OllamaConfig, OllamaProvider
from core.llm.types import (
    ChatMessage,
    GenerationRequest,
    ModelInfo,
    Role,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_response(
    status_code: int = 200,
    json_data: Any = None,
    text: str = "",
    stream_lines: list[str] | None = None,
) -> httpx.Response:
    """Construct an httpx.Response for use in a fake transport."""
    if stream_lines is not None:
        # httpx.Response is not directly streamable, but the transport below
        # uses aiter_lines from a real stream response. Construct one with
        # raw bytes for the streaming test.
        body = "\n".join(stream_lines).encode("utf-8")
        return httpx.Response(
            status_code=status_code,
            content=body,
            headers={"content-type": "application/x-ndjson"},
        )
    if json_data is not None:
        return httpx.Response(
            status_code=status_code,
            json=json_data,
        )
    return httpx.Response(
        status_code=status_code,
        text=text or json.dumps({"error": "no body"}),
    )


def make_transport(handler) -> httpx.MockTransport:
    """Build a MockTransport that delegates to ``handler(request)``."""
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestOllamaProviderConstruction:
    def test_constructs_with_defaults(self) -> None:
        provider = OllamaProvider()
        assert provider.name == "ollama"
        assert provider.default_model is None  # empty string coerced to None
        # Cleanup is async; we don't await in this test as we're only
        # verifying construction. The MockTransport is unused here.
        # close() is called in the test session via fixture.

    def test_constructs_with_kwargs(self) -> None:
        provider = OllamaProvider(
            base_url="http://localhost:11434",
            default_model="llama3",
            request_timeout_seconds=60,
        )
        assert provider.default_model == "llama3"

    def test_refuses_non_loopback_base_url(self) -> None:
        with pytest.raises(ConfigurationError, match="loopback"):
            OllamaProvider(base_url="http://10.0.0.5:11434")

    def test_refuses_public_host(self) -> None:
        with pytest.raises(ConfigurationError, match="loopback"):
            OllamaProvider(base_url="https://api.ollama.example.com")


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestOllamaProviderGenerate:
    @pytest.mark.asyncio
    async def test_successful_generate(self) -> None:
        chat_payload = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "Hello, world!"},
            "done": True,
            "done_reason": "stop",
        }
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return make_response(200, json_data=chat_payload)

        provider = OllamaProvider(default_model="llama3")
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            request = GenerationRequest(messages=[ChatMessage.user("Hi")])
            response = await provider.generate(request)
        finally:
            await provider.close()

        assert response.content == "Hello, world!"
        assert response.model == "llama3"
        assert response.finish_reason == "stop"
        # The handler was called with /api/chat and a JSON body containing the model.
        assert len(captured) == 1
        assert captured[0].url.path == "/api/chat"
        body = json.loads(captured[0].content)
        assert body["model"] == "llama3"
        assert body["stream"] is False
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_request_model_override(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return make_response(
                200,
                json_data={
                    "model": "mistral",
                    "message": {"role": "assistant", "content": "ok"},
                },
            )

        provider = OllamaProvider(default_model="llama3")
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            response = await provider.generate(
                GenerationRequest(messages=[ChatMessage.user("Hi")], model="mistral")
            )
        finally:
            await provider.close()

        assert captured[0]["model"] == "mistral"
        assert response.model == "mistral"

    @pytest.mark.asyncio
    async def test_passes_options(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return make_response(
                200,
                json_data={
                    "model": "llama3",
                    "message": {"role": "assistant", "content": "ok"},
                },
            )

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            await provider.generate(
                GenerationRequest(
                    messages=[ChatMessage.user("Hi")],
                    model="llama3",
                    temperature=0.5,
                    top_p=0.8,
                    max_tokens=64,
                    stop=["END"],
                )
            )
        finally:
            await provider.close()

        opts = captured[0]["options"]
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.8
        assert opts["num_predict"] == 64
        assert opts["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_no_model_specified_raises(self) -> None:
        provider = OllamaProvider()
        try:
            with pytest.raises(ConfigurationError, match="No model"):
                await provider.generate(
                    GenerationRequest(messages=[ChatMessage.user("Hi")])
                )
        finally:
            await provider.close()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestOllamaProviderErrors:
    @pytest.mark.asyncio
    async def test_connect_error_maps_to_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            with pytest.raises(ProviderUnavailableError):
                await provider.generate(
                    GenerationRequest(
                        messages=[ChatMessage.user("Hi")], model="llama3"
                    )
                )
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_read_timeout_maps_to_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            with pytest.raises(ProviderTimeoutError):
                await provider.generate(
                    GenerationRequest(
                        messages=[ChatMessage.user("Hi")], model="llama3"
                    )
                )
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_connect_timeout_maps_to_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("Connect timed out")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            with pytest.raises(ProviderUnavailableError):
                await provider.generate(
                    GenerationRequest(
                        messages=[ChatMessage.user("Hi")], model="llama3"
                    )
                )
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_404_maps_to_model_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(404, text="model not found")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                await provider.generate(
                    GenerationRequest(
                        messages=[ChatMessage.user("Hi")], model="nonexistent"
                    )
                )
            assert exc_info.value.model == "nonexistent"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_500_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(500, text="internal error")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            with pytest.raises(ProviderError):
                await provider.generate(
                    GenerationRequest(
                        messages=[ChatMessage.user("Hi")], model="llama3"
                    )
                )
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_malformed_json_maps_to_response_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(200, text="not json {")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            with pytest.raises(ProviderResponseError):
                await provider.generate(
                    GenerationRequest(
                        messages=[ChatMessage.user("Hi")], model="llama3"
                    )
                )
        finally:
            await provider.close()


# ---------------------------------------------------------------------------
# health() and list_models()
# ---------------------------------------------------------------------------


class TestOllamaProviderHealth:
    @pytest.mark.asyncio
    async def test_health_reachable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(
                200,
                json_data={
                    "models": [
                        {"name": "llama3:8b", "size": 5_000_000_000},
                        {"name": "mistral:7b", "size": 4_500_000_000},
                    ]
                },
            )

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            health = await provider.health()
        finally:
            await provider.close()

        assert health.reachable is True
        assert health.model_count == 2
        assert health.error is None

    @pytest.mark.asyncio
    async def test_health_unreachable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            health = await provider.health()
        finally:
            await provider.close()

        assert health.reachable is False
        assert health.model_count == 0
        assert health.error is not None

    @pytest.mark.asyncio
    async def test_list_models_parses_details(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return make_response(
                200,
                json_data={
                    "models": [
                        {
                            "name": "llama3:8b",
                            "size": 5_000_000_000,
                            "modified_at": "2024-01-01T00:00:00Z",
                            "details": {
                                "family": "llama",
                                "parameter_size": "8B",
                                "quantization_level": "Q4_0",
                            },
                        }
                    ]
                },
            )

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=make_transport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            models = await provider.list_models()
        finally:
            await provider.close()

        assert len(models) == 1
        m = models[0]
        assert isinstance(m, ModelInfo)
        assert m.name == "llama3:8b"
        assert m.family == "llama"
        assert m.parameter_size == "8B"
        assert m.quantization_level == "Q4_0"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestOllamaProviderStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self) -> None:
        lines = [
            json.dumps(
                {
                    "model": "llama3",
                    "message": {"role": "assistant", "content": "Hello"},
                    "done": False,
                }
            ),
            json.dumps(
                {
                    "model": "llama3",
                    "message": {"role": "assistant", "content": ", world"},
                    "done": False,
                }
            ),
            json.dumps(
                {
                    "model": "llama3",
                    "message": {"role": "assistant", "content": "!"},
                    "done": True,
                    "done_reason": "stop",
                }
            ),
        ]

        async def handler(request: httpx.Request) -> httpx.Response:
            return make_response(200, stream_lines=lines)

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(  # type: ignore[assignment]
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:11434",
        )
        try:
            request = GenerationRequest(messages=[ChatMessage.user("Hi")], model="llama3")
            chunks = []
            async for token in provider.stream(request):
                chunks.append(token)
        finally:
            await provider.close()

        assert chunks == ["Hello", ", world", "!"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_ollama_is_registered_by_default(self) -> None:
        from core.llm.registry import default_registry

        assert default_registry.has("ollama")

    def test_unknown_provider_raises(self) -> None:
        from core.llm.registry import default_registry

        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            default_registry.create("not-a-real-provider")

    def test_create_ollama_provider(self) -> None:
        from core.llm.registry import default_registry
        from core.llm.providers.ollama import OllamaProvider

        provider = default_registry.create(
            "ollama",
            base_url="http://127.0.0.1:11434",
            default_model="llama3",
        )
        assert isinstance(provider, OllamaProvider)
