"""Tests for Phase 2 task execution, deduplication, real-time status, and cancellation."""
import asyncio
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.v1.tasks_router import task_store


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_run_agent(monkeypatch):
    async def fake_run(task_id, task_description, attachments=None):
        pass
    monkeypatch.setattr(task_store, "run_or_enqueue", fake_run)


def test_submit_task_and_deduplication(client):
    prompt = "Unique prompt for deduplication test " + str(asyncio.get_event_loop_policy())
    payload = {"task": prompt}

    # First submission -> 201 Created
    res1 = client.post("/api/v1/tasks", json=payload)
    assert res1.status_code == 201
    data1 = res1.json()
    assert "id" in data1
    assert data1["status"] in ("queued", "running")
    assert "phase" in data1
    assert "phaseLabel" in data1

    # Immediate second identical submission -> 200 OK with same ID
    res2 = client.post("/api/v1/tasks", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["id"] == data1["id"]


def test_get_task_live_fields(client):
    prompt = "Test live status inspection"
    res = client.post("/api/v1/tasks", json={"task": prompt})
    task_id = res.json()["id"]

    res_get = client.get(f"/api/v1/tasks/{task_id}")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["id"] == task_id
    assert "status" in data
    assert "phase" in data
    assert "phaseLabel" in data
    assert "elapsedSeconds" in data
    assert isinstance(data["elapsedSeconds"], (int, float))
    assert len(data["messages"]) >= 1
    assert data["messages"][0]["content"] == prompt


def test_cancel_task(client):
    prompt = "Task to be cancelled"
    res = client.post("/api/v1/tasks", json={"task": prompt})
    task_id = res.json()["id"]

    # Cancel the task
    res_cancel = client.post(f"/api/v1/tasks/{task_id}/cancel")
    assert res_cancel.status_code == 200
    data = res_cancel.json()
    assert data["status"] == "cancelled"
    assert "cancelled" in data["phase"].lower()

    # Verify get_task also reports cancelled
    res_get = client.get(f"/api/v1/tasks/{task_id}")
    assert res_get.status_code == 200
    assert res_get.json()["status"] == "cancelled"


def test_cancel_nonexistent_task(client):
    res = client.post("/api/v1/tasks/nonexistent-uuid-1234/cancel")
    assert res.status_code == 404
