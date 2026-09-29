"""Tests for the health endpoint and other public endpoints."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app


@pytest.fixture
def client() -> AsyncClient:
    """Return an httpx AsyncClient wired to the FastAPI ASGI app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_returns_healthy(client: AsyncClient) -> None:
    """The /health endpoint should respond with status='healthy'."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "service" in body
    assert "version" in body
    assert "timestamp" in body


@pytest.mark.asyncio
async def test_health_response_shape(client: AsyncClient) -> None:
    """The /health response should contain only the expected fields."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"status", "service", "version", "timestamp"}


@pytest.mark.asyncio
async def test_info_endpoint(client: AsyncClient) -> None:
    """The /info endpoint should respond with service metadata."""
    response = await client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert "name" in body
    assert "version" in body
    assert "description" in body
    assert "endpoints" in body
    assert body["endpoints"]["health"] == "/health"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """The / endpoint should return a small descriptor payload."""
    response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "name" in body
    assert "version" in body
    assert body["health"] == "/health"


@pytest.mark.asyncio
async def test_unknown_route_returns_404(client: AsyncClient) -> None:
    """Requests to unknown routes should return 404."""
    response = await client.get("/does-not-exist")
    assert response.status_code == 404
