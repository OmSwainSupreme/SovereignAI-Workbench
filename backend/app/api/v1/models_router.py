"""Model-related routes for the v1 API.

Currently exposes only the diagnostic ``/models/status`` endpoint, which
reports whether the configured local model provider is reachable.

The endpoint is read-only, accepts no request body, and never sends prompts
or document content to any external service.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from backend.app.models.schemas import ModelStatusResponse
from backend.app.services.model_service import ModelService, get_model_service

_logger = logging.getLogger("sovereign-ai.api.models")

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Get the reachability status of the configured model provider",
    responses={
        200: {"description": "Provider is reachable."},
        503: {"description": "Provider is not currently reachable."},
    },
)
async def get_model_status(
    service: ModelService = Depends(get_model_service),
) -> JSONResponse:
    """Return the reachability status of the configured model provider.

    The endpoint calls the provider's health check (currently ``GET /api/tags``
    on Ollama). It does NOT call any generative endpoint and does NOT
    transmit prompts or other user data.
    """
    health = await service.get_status()
    body = ModelStatusResponse(
        provider=health.provider,
        reachable=health.reachable,
        model_count=health.model_count,
        error=health.error,
    )
    http_status = (
        status.HTTP_200_OK if health.reachable else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=http_status, content=body.model_dump())
