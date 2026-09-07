"""HTTP route definitions."""
from __future__ import annotations

from fastapi import APIRouter, status

from backend.app.api.v1 import v1_router
from backend.app.core.config import settings
from backend.app.models.schemas import HealthResponse, InfoResponse, utc_now_iso

router = APIRouter()

# Include the versioned API under /api/v1.
router.include_router(v1_router)


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Root endpoint returning basic app info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "health": "/health",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
)
def health() -> HealthResponse:
    """Return a simple JSON object indicating the service is healthy."""
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=utc_now_iso(),
    )


@router.get(
    "/info",
    response_model=InfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Service information",
)
def info() -> InfoResponse:
    """Return basic service information."""
    return InfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        description=(
            "Air-gapped, self-hosted, multimodal agentic AI workbench. "
            "Phase 2B adds a local Ollama provider via the Model Gateway."
        ),
        endpoints={
            "health": "/health",
            "info": "/info",
            "docs": "/docs",
            "redoc": "/redoc",
            "models_status": "/api/v1/models/status",
        },
    )
