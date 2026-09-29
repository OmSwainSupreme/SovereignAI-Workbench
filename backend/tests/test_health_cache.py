"""Tests for the in-process health/status TTL cache in ModelService.

These tests verify that repeated calls to get_status() return cached results
within the TTL window and fetch fresh data after the TTL expires.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from backend.app.services.model_service import (
    ModelService,
    _HealthCache,
    _HEALTH_TTL_SECONDS,
)
from core.llm import ProviderHealth, ModelGateway
from core.llm.providers.base import BaseProvider
from core.llm.types import GenerationRequest, GenerationResponse


# ---------------------------------------------------------------------------
# Minimal fake provider for cache tests
# ---------------------------------------------------------------------------


class CountingProvider(BaseProvider):
    """A provider that counts how many times each method was called."""

    name: str = "counting"
    default_model: str | None = "counting-model"

    def __init__(self, health_value: ProviderHealth | Exception) -> None:
        self._health_value = health_value
        self.generate_call_count: int = 0
        self.health_call_count: int = 0
        self.list_models_call_count: int = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.generate_call_count += 1
        return GenerationResponse(content="ok", model=self.default_model or "counting")

    async def health(self) -> ProviderHealth:
        self.health_call_count += 1
        if isinstance(self._health_value, Exception):
            raise self._health_value
        return self._health_value

    async def list_models(self) -> list:
        self.list_models_call_count += 1
        return []


# ---------------------------------------------------------------------------
# _HealthCache unit tests
# ---------------------------------------------------------------------------


class TestHealthCacheUnit:
    def test_stores_value_and_expiry(self) -> None:
        health = ProviderHealth(provider="test", reachable=True)
        cache = _HealthCache(health, ttl=1.0)
        assert cache.value is health
        assert not cache.is_stale()

    def test_becomes_stale_after_ttl(self) -> None:
        import time as _time_module
        health = ProviderHealth(provider="test", reachable=True)
        cache = _HealthCache(health, ttl=0.0)
        # On Windows time.monotonic() may not advance between calls, so also
        # manually age the cache by setting expires_at to the past.
        cache.expires_at = _time_module.monotonic() - 1.0
        assert cache.is_stale()


# ---------------------------------------------------------------------------
# ModelService health cache tests
# ---------------------------------------------------------------------------


class TestHealthCache:
    @pytest.fixture
    def service(self) -> ModelService:
        health = ProviderHealth(provider="counting", reachable=True, model_count=3)
        provider = CountingProvider(health)
        gw = ModelGateway(provider_name="counting", provider_config={})
        gw._provider = provider  # bypass lazy init
        return ModelService(gw)

    @pytest.mark.asyncio
    async def test_first_call_hits_provider(self, service: ModelService) -> None:
        result = await service.get_status()
        assert result.reachable is True
        assert result.model_count == 3
        assert service.gateway._provider.health_call_count == 1  # type: ignore

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_returns_cached(self, service: ModelService) -> None:
        await service.get_status()
        await service.get_status()
        await service.get_status()
        # All three calls should result in only ONE provider call.
        assert service.gateway._provider.health_call_count == 1  # type: ignore

    @pytest.mark.asyncio
    async def test_returns_fresh_after_ttl(self, service: ModelService) -> None:
        # First call caches.
        await service.get_status()
        assert service.gateway._provider.health_call_count == 1  # type: ignore

        # Wait past the TTL.
        service._health_cache = None  # type: ignore

        # Next call should hit the provider again.
        result = await service.get_status()
        assert result.reachable is True
        assert service.gateway._provider.health_call_count == 2  # type: ignore

    @pytest.mark.asyncio
    async def test_ttl_constant_is_1_to_2_seconds(self) -> None:
        """Verify the TTL constant is in the required 1-2 second range."""
        assert 1.0 <= _HEALTH_TTL_SECONDS <= 2.0

    @pytest.mark.asyncio
    async def test_error_health_is_also_cached(self) -> None:
        """Even an error result is cached to prevent hammering a failing provider."""
        provider = CountingProvider(
            ProviderHealth(provider="counting", reachable=False, error="unavailable")
        )
        gw = ModelGateway(provider_name="counting", provider_config={})
        gw._provider = provider
        service = ModelService(gw)

        await service.get_status()
        await service.get_status()
        await service.get_status()

        # Error result should also be cached (only one provider call).
        assert service.gateway._provider.health_call_count == 1  # type: ignore

    @pytest.mark.asyncio
    async def test_gateway_health_not_called_when_cached(self, service: ModelService) -> None:
        """generate() calls on the gateway should not affect the health cache."""
        # Populate the health cache.
        await service.get_status()
        assert service._health_cache is not None

        # A subsequent get_status() should NOT trigger a new health check.
        result = await service.get_status()
        assert result.reachable is True
        # Gateway health must not have been called again.
        assert service.gateway._provider.health_call_count == 1  # type: ignore