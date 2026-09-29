"""Knowledge-related routes for the v1 API.

Exposes endpoints for RAG search operations.
All operations go through the search_knowledge_base tool and policy engine.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from core.agent.types import ToolCall
from core.rag.knowledge_base import create_knowledge_base
from core.rag.tools import _search_kb_callable
from core.security.policy_engine import get_policy_engine
from core.tools.workspace import Workspace

logger = logging.getLogger("sovereign-ai.api.knowledge")

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query in natural language.")
    top_k: Optional[int] = Field(None, description="Maximum number of chunks to return (default: 5).")
    min_score: Optional[float] = Field(None, description="Minimum cosine similarity threshold, 0.0 to 1.0.")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    total_available: int
    results: List[SearchResultItem]
    count: int


class CollectionsResponse(BaseModel):
    """Response indicating collection management status.

    The RAG knowledge base supports document ingestion and search, but
    does not currently support collection management operations (create,
    list, delete collections). This endpoint reports that limitation
    transparently rather than faking persistence.
    """
    collections_supported: bool = False
    message: str = (
        "Collection management is not currently supported by the RAG core. "
        "Use the search_knowledge_base endpoint to query ingested documents."
    )


def get_workspace() -> Workspace:
    """Return a workspace instance for file operations."""
    workspace_path = settings.workspace_path or "workspace"
    return Workspace(workspace_path)


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(
    request: SearchRequest,
) -> SearchResponse:
    """Search the local knowledge base for relevant document chunks."""
    workspace = get_workspace()
    policy_engine = get_policy_engine()
    knowledge_base = create_knowledge_base(workspace)

    # Create a tool call for search_knowledge_base
    tool_call = ToolCall(
        call_id="search_knowledge_base_call",
        tool_name="search_knowledge_base",
        arguments={
            "query": request.query,
            "top_k": request.top_k or 5,
            "min_score": request.min_score or 0.0,
        },
    )

    # Check policy
    policy_decision = policy_engine.evaluate(tool_call)
    if not policy_decision.is_allowed():
        logger.warning("Policy denied search_knowledge_base: %s", policy_decision.reason)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=policy_decision.reason,
        )

    # Execute the tool
    try:
        result = await _search_kb_callable(knowledge_base)(tool_call.arguments)
        # The result should be a dict with query, total_available, results, count
        return SearchResponse(**result)
    except Exception as exc:
        logger.exception("Error searching knowledge base: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search knowledge base",
        )


@router.get("/collections", response_model=CollectionsResponse)
async def get_collections() -> CollectionsResponse:
    """Report collection management status.

    The RAG knowledge base supports document ingestion and search, but
    does not currently support collection management operations (create,
    list, delete collections). This endpoint reports that limitation
    transparently rather than faking persistence.

    Returns:
        A CollectionsResponse indicating that collections are not supported,
        with a helpful message for the frontend.
    """
    return CollectionsResponse()