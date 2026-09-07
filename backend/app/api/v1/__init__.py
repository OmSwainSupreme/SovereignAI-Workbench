"""Versioned v1 API routes."""
from fastapi import APIRouter

from backend.app.api.v1.models_router import router as models_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(models_router)

__all__ = ["v1_router"]
