"""Tests for httpx exception translation in the Ollama provider.

These verify that the provider boundary catches the full httpx exception
hierarchy (not just the common cases) and converts unknown transport-level
errors into ``ProviderError`` rather than letting raw ``httpx.*`` exceptions
leak through the Model Gateway.

A fake httpx transport is used; no live Ollama server is required.
"""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from core.llm import (
    ProviderError,
    ProviderUnavailableError,
)
from core.llm.providers.ollama import OllamaProvider
from core.llm.types import ChatMessage, GenerationRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaProvider:
    """Build an OllamaProvider whose httpx client is wired to a fake transport."""
    provider = OllamaProvider(default_model="llama3")
    provider._client = httpx.AsyncClient(  # type: ignore[assignment]
        transport=httpx.MockTransport(handler),
        base_url="http://127.0.0.1:11434",
    )
    return provider


def _ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "model": "llama3",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        },
    )


def _request(request: GenerationRequest) -> GenerationRequest:
    return GenerationRequest(messages=[ChatMessage.user("Hello")], model="llama3")


# ---------------------------------------------------------------------------
# httpx.HTTPError translation (non-streaming path)
# ---------------------------------------------------------------------------


class TestHttpxErrorTranslationGenerate:
    @pytest.mark.asyncio
    async def test_remote_protocol_error_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("server closed connection without response")

        provider = _make_provider(handler)
        try:
            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_request(GenerationRequest(
                    messages=[ChatMessage.user("Hello")], model="llama3"
                )))
            # Original httpx exception is chained.
            assert isinstance(exc_info.value.__cause__, httpx.RemoteProtocolError)
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_network_error_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.NetworkError("connection reset")

        provider = _make_provider(handler)
        try:
            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(GenerationRequest(
                    messages=[ChatMessage.user("Hello")], model="llama3"
                ))
            assert isinstance(exc_info.value.__cause__, httpx.NetworkError)
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_pool_timeout_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.PoolTimeout("pool exhausted")

        provider = _make_provider(handler)
        try:
            with pytest.raises(ProviderError):
                await provider.generate(GenerationRequest(
                    messages=[ChatMessage.user("Hello")], model="llama3"
                ))
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_existing_connect_error_still_maps_to_unavailable(self) -> None:
        """Regression: ConnectError still maps to ProviderUnavailableError, not ProviderError."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        provider = _make_provider(handler)
        try:
            with pytest.raises(ProviderUnavailableError):
                await provider.generate(GenerationRequest(
                    messages=[ChatMessage.user("Hello")], model="llama3"
                ))
        finally:
            await provider.close()


# ---------------------------------------------------------------------------
# httpx.HTTPError translation (streaming path)
# ---------------------------------------------------------------------------


class TestHttpxErrorTranslationStream:
    @pytest.mark.asyncio
    async def test_stream_network_error_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.NetworkError("connection dropped mid-stream")

        provider = _make_provider(handler)
        try:
            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.stream(GenerationRequest(
                    messages=[ChatMessage.user("Hello")], model="llama3"
                )):
                    pass
            assert isinstance(exc_info.value.__cause__, httpx.NetworkError)
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_stream_remote_protocol_error_maps_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("protocol error")

        provider = _make_provider(handler)
        try:
            with pytest.raises(ProviderError):
                async for _ in provider.stream(GenerationRequest(
                    messages=[ChatMessage.user("Hello")], model="llama3"
                )):
                    pass
        finally:
            await provider.close()


# ---------------------------------------------------------------------------
# Verify normal success path still works (regression)
# ---------------------------------------------------------------------------


class TestRegressionSuccess:
    @pytest.mark.asyncio
    async def test_generate_succeeds_via_translation(self) -> None:
        provider = _make_provider(_ok_response)
        try:
            response = await provider.generate(GenerationRequest(
                messages=[ChatMessage.user("Hello")], model="llama3"
            ))
            assert response.content == "ok"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_stream_succeeds_via_translation(self) -> None:
        lines = [
            json.dumps({"model": "llama3", "message": {"role": "assistant", "content": "hi"}, "done": False}),
            json.dumps({"model": "llama3", "message": {"role": "assistant", "content": "!"}, "done": True}),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content="\n".join(lines).encode("utf-8"),
                headers={"content-type": "application/x-ndjson"},
            )

        provider = _make_provider(handler)
        try:
            tokens = []
            async for token in provider.stream(GenerationRequest(
                messages=[ChatMessage.user("Hello")], model="llama3"
            )):
                tokens.append(token)
            assert tokens == ["hi", "!"]
        finally:
            await provider.close()