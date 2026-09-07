"""Pydantic request/response models for the API."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str = Field(..., description="Service health status.")
    service: str = Field(..., description="Service display name.")
    version: str = Field(..., description="Service version string.")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of the response.")


class InfoResponse(BaseModel):
    """Response model for the /info endpoint."""

    name: str = Field(..., description="Service display name.")
    version: str = Field(..., description="Service version string.")
    description: str = Field(..., description="Service description.")
    endpoints: dict[str, str] = Field(..., description="Available endpoint paths.")


# ---------------------------------------------------------------------------
# Model service schemas
# ---------------------------------------------------------------------------


class ModelStatusResponse(BaseModel):
    """Response model for the /api/v1/models/status endpoint.

    Reports whether the configured local model provider is reachable.
    """

    provider: str = Field(..., description="Configured provider name.")
    reachable: bool = Field(..., description="Whether the provider backend is reachable.")
    model_count: int = Field(
        default=0,
        description="Number of models currently available on the provider.",
    )
    error: Optional[str] = Field(
        default=None,
        description=(
            "Error message if the provider is unreachable. "
            "Absent when reachable=True."
        ),
    )


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
