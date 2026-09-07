"""Tests for the ModelGateway and ProviderRegistry."""
from __future__ import annotations

import logging

import pytest

from core.llm import (
    ConfigurationError,
    GenerationRequest,
    GenerationResponse,
    ModelGateway,
    ModelInfo,
    ProviderHealth,
    ProviderUnavailableError,
)
from core.llm.errors import LLMError
from core.llm.providers.base import BaseProvider
from core.llm.registry import ProviderRegistry, default_registry
from core.llm.types import ChatMessage, Role


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    """A minimal provider stub used for gateway tests."""

    name: str = "fake"
    default_model: str | None = "fake-model"

    def __init__(
        self,
        generate_response: GenerationResponse | Exception = None,
        stream_tokens: list[str] | Exception = None,
        health_response: ProviderHealth | Exception = None,
        list_models_response: list[ModelInfo] | Exception = None,
    ) -> None:
        self._gen_resp = generate_response
        self._stream = stream_tokens
        self._health = health_response
        self._list_models = list_models_response
        self.generate_called_with: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.generate_called_with.append(request)
        if isinstance(self._gen_resp, Exception):
            raise self._gen_resp
        return self._gen_resp

    async def stream(self, request: GenerationRequest):
        if isinstance(self._stream, Exception):
            raise self._stream
        for token in self._stream:
            yield token

    async def health(self) -> ProviderHealth:
        if isinstance(self._health, Exception):
            raise self._health
        return self._health

    async def list_models(self) -> list[ModelInfo]:
        if isinstance(self._list_models, Exception):
            raise self._list_models
        return self._list_models


# ---------------------------------------------------------------------------
# Gateway construction
# ---------------------------------------------------------------------------


class TestModelGatewayConstruction:
    def test_stores_provider_name(self) -> None:
        gw = ModelGateway(provider_name="fake", provider_config={})
        assert gw.provider_name == "fake"

    def test_provider_not_instantiated_until_first_call(self) -> None:
        registry = ProviderRegistry()
        registry.register("fake", FakeProvider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)
        assert gw._provider is None


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGatewayGenerate:
    @pytest.mark.asyncio
    async def test_delegates_to_provider(self) -> None:
        resp = GenerationResponse(content="42", model="fake-model")
        registry = ProviderRegistry()
        registry.register("fake", lambda: FakeProvider(generate_response=resp))
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        request = GenerationRequest(messages=[ChatMessage.user("What is 2+2?")])
        result = await gw.generate(request)

        assert result.content == "42"

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self) -> None:
        provider = FakeProvider(
            generate_response=ProviderUnavailableError("Connection refused")
        )
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        with pytest.raises(ProviderUnavailableError):
            await gw.generate(GenerationRequest(messages=[ChatMessage.user("Hello")]))

    @pytest.mark.asyncio
    async def test_non_llm_error_propagates(self) -> None:
        """Non-LLMError exceptions (e.g. ValueError) propagate unchanged."""
        provider = FakeProvider(generate_response=ValueError("boom"))
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        with pytest.raises(ValueError, match="boom"):
            await gw.generate(GenerationRequest(messages=[ChatMessage.user("Hello")]))

    @pytest.mark.asyncio
    async def test_logs_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        resp = GenerationResponse(content="answer", model="fake-model")
        registry = ProviderRegistry()
        registry.register("fake", lambda: FakeProvider(generate_response=resp))
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        await gw.generate(GenerationRequest(messages=[ChatMessage.user("Hello")]))

        gw_records = [r.message for r in caplog.records if r.name == "sovereign-ai.gateway"]
        assert any("generate.start" in r for r in gw_records)
        assert any("generate.done" in r for r in gw_records)
        # Verify no prompt or response content appears in any log record.
        for msg in gw_records:
            assert "Hello" not in msg, f"Prompt leaked into log: {msg!r}"
            assert "answer" not in msg, f"Response leaked into log: {msg!r}"

    @pytest.mark.asyncio
    async def test_logs_error_on_llm_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        provider = FakeProvider(
            generate_response=ProviderUnavailableError("Connection refused")
        )
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        with pytest.raises(ProviderUnavailableError):
            await gw.generate(GenerationRequest(messages=[ChatMessage.user("Hello")]))

        records = [r.message for r in caplog.records]
        assert any("generate.fail" in r for r in records)
        assert any("ProviderUnavailableError" in r for r in records)
        # Error type must NOT be "unknown" - it must be the actual exception name.
        fail_records = [r for r in records if "generate.fail" in r]
        assert fail_records, "generate.fail log record not found"
        for rec in fail_records:
            assert "ProviderUnavailableError" in rec, f"Expected ProviderUnavailableError in: {rec!r}"

    @pytest.mark.asyncio
    async def test_logs_error_on_non_llm_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Non-LLMError exceptions must still log their correct type, not 'unknown'."""
        caplog.set_level(logging.WARNING)
        provider = FakeProvider(generate_response=ValueError("boom"))
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        with pytest.raises(ValueError):
            await gw.generate(GenerationRequest(messages=[ChatMessage.user("Hello")]))

        records = [r.message for r in caplog.records]
        fail_records = [r for r in records if "generate.fail" in r]
        assert fail_records, "generate.fail log record not found"
        # Must log the actual exception type, not "unknown".
        for rec in fail_records:
            assert "ValueError" in rec, f"Expected ValueError in fail log, got: {rec!r}"
            assert "unknown" not in rec.lower(), f"Got 'unknown' error type in: {rec!r}"


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestGatewayStream:
    @pytest.mark.asyncio
    async def test_delegates_stream(self) -> None:
        provider = FakeProvider(stream_tokens=["a", "b", "c"])
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        tokens = [
            token
            async for token in gw.stream(
                GenerationRequest(messages=[ChatMessage.user("Hi")])
            )
        ]
        assert tokens == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self) -> None:
        provider = FakeProvider(stream_tokens=ProviderUnavailableError("Gone"))
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        with pytest.raises(LLMError):
            [
                token
                async for token in gw.stream(
                    GenerationRequest(messages=[ChatMessage.user("Hi")])
                )
            ]

    @pytest.mark.asyncio
    async def test_non_llm_error_propagates(self) -> None:
        """Non-LLMError exceptions from a stream must propagate unchanged."""
        provider = FakeProvider(stream_tokens=RuntimeError("stream broken"))
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        with pytest.raises(RuntimeError, match="stream broken"):
            [
                token
                async for token in gw.stream(
                    GenerationRequest(messages=[ChatMessage.user("Hi")])
                )
            ]

    @pytest.mark.asyncio
    async def test_logs_error_on_stream_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        provider = FakeProvider(stream_tokens=ProviderUnavailableError("Gone"))
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)

        with pytest.raises(LLMError):
            [
                token
                async for token in gw.stream(
                    GenerationRequest(messages=[ChatMessage.user("Hi")])
                )
            ]

        records = [r.message for r in caplog.records]
        fail_records = [r for r in records if "stream.fail" in r]
        assert fail_records, "stream.fail log record not found"
        for rec in fail_records:
            assert "ProviderUnavailableError" in rec, f"Expected ProviderUnavailableError in: {rec!r}"


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


class TestGatewayHealth:
    @pytest.mark.asyncio
    async def test_delegates_health(self) -> None:
        health = ProviderHealth(provider="fake", reachable=True, model_count=3)
        registry = ProviderRegistry()
        registry.register("fake", FakeProvider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)
        # Inject the fake directly so we bypass provider construction.
        gw._provider = FakeProvider(health_response=health)

        result = await gw.health()
        assert result.reachable is True
        assert result.model_count == 3

    @pytest.mark.asyncio
    async def test_health_error_raises(self) -> None:
        registry = ProviderRegistry()
        registry.register("fake", FakeProvider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)
        gw._provider = FakeProvider(health_response=ProviderUnavailableError("Gone"))

        with pytest.raises(LLMError):
            await gw.health()


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


class TestGatewayListModels:
    @pytest.mark.asyncio
    async def test_delegates_list_models(self) -> None:
        models = [
            ModelInfo(name="fake-model"),
            ModelInfo(name="fake-model-2"),
        ]
        registry = ProviderRegistry()
        registry.register("fake", FakeProvider)
        gw = ModelGateway(provider_name="fake", provider_config={}, registry=registry)
        gw._provider = FakeProvider(list_models_response=models)

        result = await gw.list_models()
        assert len(result) == 2
        assert result[0].name == "fake-model"


# ---------------------------------------------------------------------------
# ProviderRegistry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_register_and_create(self) -> None:
        registry = ProviderRegistry()
        registry.register("test", FakeProvider)
        assert registry.has("test")
        provider = registry.create("test")
        assert isinstance(provider, FakeProvider)

    def test_duplicate_registration_replaces(self) -> None:
        registry = ProviderRegistry()
        registry.register("test", FakeProvider)

        class OtherProvider(BaseProvider):
            name = "other"

            async def generate(self, request):
                return GenerationResponse(content="x", model="other")

            async def health(self):
                return ProviderHealth(provider="other", reachable=True)

            async def list_models(self):
                return []

        registry.register("test", OtherProvider)
        provider = registry.create("test")
        assert isinstance(provider, OtherProvider)

    def test_unregister(self) -> None:
        registry = ProviderRegistry()
        registry.register("test", FakeProvider)
        registry.unregister("test")
        assert not registry.has("test")

    def test_unknown_name_raises(self) -> None:
        registry = ProviderRegistry()
        with pytest.raises(ConfigurationError):
            registry.create("does-not-exist")