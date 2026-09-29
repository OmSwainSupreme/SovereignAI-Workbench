"""Tests for the files API endpoints (list, upload, download).

These cover the real integration contract the frontend depends on: multipart
upload, listing uploaded files in the frontend-facing shape, download by
handle, and clean rejection of unsupported content / policy denials.
"""
from __future__ import annotations

import shutil
import tempfile

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.core.config import settings
from backend.app.main import app
from core.security.policy_engine import DENY, init_policy_engine, reset_policy_engine


@pytest.fixture
def workspace(monkeypatch) -> str:
    """Point the files router at an isolated temp workspace with policy installed."""
    ws = tempfile.mkdtemp(prefix="sovereign_ai_files_api_")
    monkeypatch.setattr(settings, "workspace_path", ws)
    init_policy_engine({})  # file_system.read/write/list ALLOW by default
    yield ws
    reset_policy_engine()
    shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture
def client(workspace) -> AsyncClient:
    """Return an httpx AsyncClient wired to the FastAPI ASGI app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_upload_then_list_then_download(client: AsyncClient) -> None:
    """Upload a text file, see it listed, and download the exact bytes."""
    content = b"SovereignAI demo content line one.\nSecond line.\n"
    upload = await client.post(
        "/api/v1/files/upload",
        files={"files": ("demo.txt", content, "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["message"] == "Successfully uploaded 1 file(s)"
    assert body["uploaded"] == ["demo.txt"]

    # The uploaded file must appear on the list endpoint.
    listing = await client.get("/api/v1/files")
    assert listing.status_code == 200
    files = listing.json()["files"]
    assert len(files) == 1
    entry = files[0]
    assert entry["name"] == "demo.txt"
    assert entry["kind"] == "file"
    assert entry["origin"] == "user_upload"
    assert entry["downloadable"] is True
    assert entry["handle"] == "demo.txt"
    assert entry["sizeBytes"] == len(content)

    # Download by handle returns the exact content.
    download = await client.get("/api/v1/files/demo.txt")
    assert download.status_code == 200
    assert download.content == content


@pytest.mark.asyncio
async def test_multiple_uploads_appear_in_listing(client: AsyncClient) -> None:
    """Uploading several files lists all of them."""
    resp = await client.post(
        "/api/v1/files/upload",
        files=[
            ("files", ("a.txt", b"a", "text/plain")),
            ("files", ("b.md", b"b", "text/markdown")),
        ],
    )
    assert resp.status_code == 200
    assert set(resp.json()["uploaded"]) == {"a.txt", "b.md"}

    listing = await client.get("/api/v1/files")
    names = {f["name"] for f in listing.json()["files"]}
    assert names == {"a.txt", "b.md"}


@pytest.mark.asyncio
async def test_upload_rejects_binary_content(client: AsyncClient) -> None:
    """Non-UTF-8, non-image content is rejected with a clean client error, not a 500."""
    resp = await client.post(
        "/api/v1/files/upload",
        files={"files": ("data.bin", b"\x89\x00\x01\x02\x03", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"].lower()

    # Nothing was written to the workspace.
    listing = await client.get("/api/v1/files")
    assert listing.json()["files"] == []


@pytest.mark.asyncio
async def test_upload_accepts_image_content(client: AsyncClient) -> None:
    """Image files (PNG, JPG, JPEG) are accepted for multimodal tasks."""
    resp = await client.post(
        "/api/v1/files/upload",
        files={"files": ("img.png", b"\x89PNG\r\n\x1a\n\x00\x00", "image/png")},
    )
    assert resp.status_code == 200
    assert "img.png" in resp.json()["uploaded"]

    listing = await client.get("/api/v1/files")
    assert any(f["name"] == "img.png" for f in listing.json()["files"])


@pytest.mark.asyncio
async def test_upload_denied_by_policy(client: AsyncClient) -> None:
    """A policy that denies write_file must block the upload with 403."""
    init_policy_engine({"file_system": {"write_file": DENY}})
    resp = await client.post(
        "/api/v1/files/upload",
        files={"files": ("secret.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_empty_workspace(client: AsyncClient) -> None:
    """An empty workspace lists zero files."""
    resp = await client.get("/api/v1/files")
    assert resp.status_code == 200
    assert resp.json()["files"] == []


@pytest.mark.asyncio
async def test_download_missing_handle_returns_404(client: AsyncClient) -> None:
    """An unknown handle returns 404, not an error."""
    resp = await client.get("/api/v1/files/does-not-exist.txt")
    assert resp.status_code == 404