"""RAG tool for the agent tool registry (Phase 5B).

This module provides the ``search_knowledge_base`` tool definition and its
callable executor, wired into the :class:`DefaultToolRegistry` via
:func:`register_rag_tools`.

The tool follows the same patterns as :mod:`core.tools.registry`:
a :class:`core.agent.interfaces.ToolDefinition` plus a callable bridge.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from core.agent.errors import ToolExecutionError
from core.agent.interfaces import ToolDefinition
from core.agent.registry import DefaultToolRegistry
from core.agent.types import ToolCall, ToolResult
from core.rag.knowledge_base import KnowledgeBase


_logger = logging.getLogger("sovereign-ai.rag.tools")


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

SEARCH_KB_TOOL = ToolDefinition(
    name="search_knowledge_base",
    description=(
        "Search the local knowledge base for relevant document chunks. "
        "Returns ranked results with source metadata (filename, document, "
        "chunk index, character offsets) and similarity scores. "
        "Use this when the user asks about information from ingested "
        "documents, manuals, SOPs, reports, or any indexed knowledge."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query in natural language. "
                    "Be specific and include key terms from the question."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of chunks to return (default: 5).",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
            "min_score": {
                "type": "number",
                "description": (
                    "Minimum cosine similarity threshold, 0.0 to 1.0. "
                    "Only results above this score are returned (default: 0.0)."
                ),
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["query"],
    },
    output_description=(
        "A structured search result containing ranked document chunks, "
        "each with: chunk_id, text, metadata (document_id, filename, "
        "content_type, chunk_index, char_start, char_end), score, and rank."
    ),
    capability="knowledge",
)


# ---------------------------------------------------------------------------
# Callable bridge
# ---------------------------------------------------------------------------


def _search_kb_callable(knowledge_base: KnowledgeBase):
    """Return a sync callable that wraps the KnowledgeBase search."""

    async def _call(args: Mapping[str, Any]) -> object:
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        min_score = float(args.get("min_score", 0.0))
        # Clamp to the valid range so callers who pass values slightly
        # outside [0, 1] get graceful behaviour rather than an error.
        min_score = max(0.0, min(1.0, min_score))

        result = knowledge_base.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        # Convert to a plain serialisable dict for the agent.
        return {
            "query": result.query,
            "total_available": result.total_available,
            "results": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "text": r.chunk.text,
                    "metadata": {
                        "document_id": r.chunk.metadata.document_id,
                        "filename": r.chunk.metadata.filename,
                        "content_type": r.chunk.metadata.content_type,
                        "chunk_index": r.chunk.metadata.chunk_index,
                        "char_start": r.chunk.metadata.char_start,
                        "char_end": r.chunk.metadata.char_end,
                    },
                    "score": round(r.score, 4),
                    "rank": r.rank,
                }
                for r in result.results
            ],
            "count": len(result.results),
        }

    return _call


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def register_rag_tools(
    registry: DefaultToolRegistry,
    knowledge_base: KnowledgeBase,
) -> None:
    """Register the search_knowledge_base tool into ``registry``.

    After this call, the agent's tool registry will include the RAG tool.
    The tool is wired to the given ``knowledge_base`` instance.

    Args:
        registry: A :class:`DefaultToolRegistry` to populate.
        knowledge_base: The :class:`KnowledgeBase` to query.

    Raises:
        TypeError: if ``registry`` is not a :class:`DefaultToolRegistry`.
    """
    if not isinstance(registry, DefaultToolRegistry):
        raise TypeError(
            "register_rag_tools requires a DefaultToolRegistry "
            "(other registries do not support callables)"
        )

    registry.register(SEARCH_KB_TOOL, _search_kb_callable(knowledge_base))
    _logger.info(
        "rag.tools.registered  kb_stats=%s",
        knowledge_base.stats(),
    )
