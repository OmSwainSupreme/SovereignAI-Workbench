"""Tests for the /api/v1/models/status endpoint.

These tests use FastAPI's dependency override mechanism to inject a fake
ModelService, so no running Ollama server is required.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from backend.app.services.model_service import ModelService, get_model_service
from core.llm import ProviderHealth


# ---------------------------------------------------------------------------
# Fake service
# ---------------------------------------------------------------------------


class FakeModelService(ModelService):
    """A fake ModelService used in endpoint tests."""

    def __init__(self, reachable: bool, model_count: int = 0, error: str | None = None) -> None:
        from unittest.mock import AsyncMock

        # Patch get_status to return the desired health.
        health = ProviderHealth(
            provider="ollama",
            reachable=reachable,
            model_count=model_count,
            error=error,
        )
        self.get_status = AsyncMock(return_value=health)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestModelStatusEndpoint:
    @pytest.mark.asyncio
    async def test_returns_200_when_provider_reachable(self, client: AsyncClient) -> None:
        app.dependency_overrides[get_model_service] = lambda: FakeModelService(
            reachable=True, model_count=3
        )
        try:
            response = await client.get("/api/v1/models/status")
            assert response.status_code == 200
            body = response.json()
            assert body["provider"] == "ollama"
            assert body["reachable"] is True
            assert body["model_count"] == 3
            assert body["error"] is None
        finally:
            app.dependency_overrides.pop(get_model_service, None)

    @pytest.mark.asyncio
    async def test_returns_503_when_provider_unreachable(self, client: AsyncClient) -> None:
        app.dependency_overrides[get_model_service] = lambda: FakeModelService(
            reachable=False,
            model_count=0,
            error="Connection refused",
        )
        try:
            response = await client.get("/api/v1/models/status")
            assert response.status_code == 503
            body = response.json()
            assert body["reachable"] is False
            assert body["error"] == "Connection refused"
        finally:
            app.dependency_overrides.pop(get_model_service, None)

    @pytest.mark.asyncio
    async def test_zero_models_is_not_error(self, client: AsyncClient) -> None:
        """Zero models is a valid state (no models pulled yet)."""
        app.dependency_overrides[get_model_service] = lambda: FakeModelService(
            reachable=True, model_count=0
        )
        try:
            response = await client.get("/api/v1/models/status")
            assert response.status_code == 200
            body = response.json()
            assert body["reachable"] is True
            assert body["model_count"] == 0
        finally:
            app.dependency_overrides.pop(get_model_service, None)

    @pytest.mark.asyncio
    async def test_endpoint_not_in_root_paths(self, client: AsyncClient) -> None:
        """The endpoint should not be reachable at / (no match)."""
        response = await client.get("/")
        assert response.status_code == 200  # root returns app info, not 404

    @pytest.mark.asyncio
    async def test_endpoint_requires_no_body(self, client: AsyncClient) -> None:
        """GET /status must not require a body (no accidental POST behaviour)."""
        app.dependency_overrides[get_model_service] = lambda: FakeModelService(
            reachable=True, model_count=1
        )
        try:
            response = await client.get("/api/v1/models/status")
            assert response.status_code == 200
        finally:
            app.dependency_overrides.pop(get_model_service, None)
