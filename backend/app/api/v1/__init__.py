"""Versioned v1 API routes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.v1.activity_router import router as activity_router
from backend.app.api.v1.code_router import router as code_router
from backend.app.api.v1.files_router import router as files_router
from backend.app.api.v1.knowledge_router import router as knowledge_router
from backend.app.api.v1.models_router import router as models_router
from backend.app.api.v1.tasks_router import router as tasks_router
from backend.app.api.v1.vision_router import router as vision_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(activity_router)
v1_router.include_router(code_router)
v1_router.include_router(files_router)
v1_router.include_router(knowledge_router)
v1_router.include_router(models_router)
v1_router.include_router(tasks_router)
v1_router.include_router(vision_router)

__all__ = ["v1_router"]