"""FastAPI application entry point."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure repository root and backend directory are in sys.path across worker spawns
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
_backend_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as api_router
from backend.app.core.config import settings
from backend.app.core.logging import configure_logging
from core.security.policy_engine import init_policy_engine

# Initialise logging first so application startup messages are captured.
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "SovereignAI Workbench backend API. "
        "Phase 2B exposes a Model Gateway with a local Ollama provider, plus a "
        "diagnostic /api/v1/models/status endpoint. "
        "Future phases will add model routing, agent workflows, RAG, and more."
    ),
    debug=settings.debug,
)

# Configure CORS to allow the frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


import asyncio
import httpx


async def _warm_default_model() -> None:
    """Asynchronously warm the default general model in Ollama at startup.

    Sends a 1-token keepalive ping to Ollama so the model weights are loaded
    into CPU RAM in the background without blocking server startup or API availability.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "qwen3:4b",
                    "prompt": "hi",
                    "keep_alive": "15m",
                    "options": {"num_predict": 1},
                },
            )
            logger.info("model_warming.done model=qwen3:4b")
    except Exception as exc:
        logger.info("model_warming.skipped reason=%s", exc)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise the policy engine, warm default model, and log application startup."""
    # Initialise the security policy engine (Phase 6) with default config
    # (which loads from config/policy.yaml if present, else secure defaults).
    init_policy_engine()
    logger.info(
        "%s v%s started on %s:%d",
        settings.app_name,
        settings.app_version,
        settings.host,
        settings.port,
    )
    # Warm the default model in the background (non-blocking)
    asyncio.create_task(_warm_default_model())