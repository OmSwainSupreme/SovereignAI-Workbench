"""FastAPI application entry point."""
from __future__ import annotations

import logging

from fastapi import FastAPI

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

app.include_router(api_router)


@app.on_event("startup")
def on_startup() -> None:
    """Initialise the policy engine and log that the application has started."""
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
