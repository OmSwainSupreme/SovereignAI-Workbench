"""Regression tests for Activity API and Policy Engine integration."""
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from core.security.policy_engine import init_policy_engine, reset_policy_engine


@pytest.fixture(autouse=True)
def setup_policy():
    init_policy_engine()
    yield
    reset_policy_engine()


@pytest.mark.asyncio
async def test_activity_allowed_by_default():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/activity")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_activity_denied_when_policy_denies():
    init_policy_engine(config={"audit": {"read_audit_log": "DENY"}})
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/activity")
        assert response.status_code == 403
        data = response.json()
        assert "detail" in data
        assert "Denied by policy" in data["detail"]
