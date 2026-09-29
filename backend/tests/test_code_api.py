"""Regression tests for Code Runs API endpoints and policy enforcement."""
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
async def test_list_code_runs_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/code")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_code_run_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/code/exec-nonexistent123")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_code_execute_denied_by_default_policy():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/code/execute",
            json={"code": "print('hello')", "language": "python"},
        )
        assert response.status_code == 403
        data = response.json()
        assert "Denied by policy" in data.get("detail", "")
