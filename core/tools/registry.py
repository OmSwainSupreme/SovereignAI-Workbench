"""Convenience helpers to register file and document tools with a :class:`ToolRegistry`.

This module provides functions that register the Phase 5A file tools
(list_files, read_file, write_file) and the Phase 5E document tool
(create_document) into a :class:`DefaultToolRegistry`.

Example::

    from core.agent.registry import DefaultToolRegistry
    from core.tools import Workspace
    from core.tools.registry import register_file_tools, register_document_tools

    workspace = Workspace(root_path="/data/workspace")
    registry = DefaultToolRegistry()
    register_file_tools(registry, workspace)
    register_document_tools(registry, workspace)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

from core.agent.registry import DefaultToolRegistry
from core.tools.file_tools import (
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    FileToolExecutor,
)
from core.tools.workspace import Workspace


_logger = logging.getLogger("sovereign-ai.tools.registry")


def register_file_tools(
    registry: DefaultToolRegistry,
    workspace: Workspace,
) -> None:
    """Register the three file tools into ``registry``.

    The tools' Python callables are bound to the given ``workspace``.
    After this call, the registry can be handed to a
    :class:`core.agent.registry.SyncToolExecutor` and used by the agent.

    Args:
        registry: A :class:`DefaultToolRegistry` to populate.
        workspace: The :class:`Workspace` the tools should operate in.

    Raises:
        TypeError: if ``registry`` is not a :class:`DefaultToolRegistry`.
            (Other ToolRegistry implementations do not support callables.)
    """
    if not isinstance(registry, DefaultToolRegistry):
        raise TypeError(
            "register_file_tools requires a DefaultToolRegistry "
            "(other registries do not support callables)"
        )

    executor = FileToolExecutor(workspace)
    registry.register(LIST_FILES_TOOL, _list_files_callable(executor))
    registry.register(READ_FILE_TOOL, _read_file_callable(executor))
    registry.register(WRITE_FILE_TOOL, _write_file_callable(executor))

    _logger.info(
        "tools.registered  workspace=%s  tools=%s",
        workspace.name,
        [LIST_FILES_TOOL.name, READ_FILE_TOOL.name, WRITE_FILE_TOOL.name],
    )


def register_document_tools(
    registry: DefaultToolRegistry,
    workspace: Workspace,
) -> None:
    """Register the document generation tool into ``registry``.

    The tool's Python callable is bound to the given ``workspace``.
    After this call, the registry can be handed to a
    :class:`core.agent.registry.SyncToolExecutor` and used by the agent.

    Args:
        registry: A :class:`DefaultToolRegistry` to populate.
        workspace: The :class:`Workspace` the tools should operate in.

    Raises:
        TypeError: if ``registry`` is not a :class:`DefaultToolRegistry`.
    """
    if not isinstance(registry, DefaultToolRegistry):
        raise TypeError(
            "register_document_tools requires a DefaultToolRegistry "
            "(other registries do not support callables)"
        )

    from core.tools.document_tools import (
        CREATE_DOCUMENT_TOOL,
        DocumentToolExecutor,
    )

    executor = DocumentToolExecutor(workspace)
    registry.register(CREATE_DOCUMENT_TOOL, _create_document_callable(executor))

    _logger.info(
        "tools.registered  workspace=%s  tools=%s",
        workspace.name,
        [CREATE_DOCUMENT_TOOL.name],
    )


# ---------------------------------------------------------------------------
# Callable bridges
# ---------------------------------------------------------------------------
#
# The :class:`DefaultToolRegistry` stores callables that take a
# ``Mapping[str, Any]`` (the arguments) and return the tool's output.
# The :class:`FileToolExecutor` is async, so we wrap each tool in a
# small bridge that:
#
# 1. Constructs a synthetic :class:`ToolCall`.
# 2. Calls the executor's async method.
# 3. Returns the result's ``output`` (or raises on error).
#
# SyncToolExecutor handles both sync and async callables (it awaits
# the result if it is a coroutine), so the bridge is just a thin
# coroutine-returning function.


def _list_files_callable(executor: FileToolExecutor):
    async def _call(args: Mapping) -> object:
        from core.agent.types import ToolCall, ToolResult

        # We don't have a real call_id here; the registry's
        # SyncToolExecutor never sees it. We use a placeholder.
        call = ToolCall(
            call_id="list_files_call",
            tool_name=LIST_FILES_TOOL.name,
            arguments=dict(args),
        )
        result: ToolResult = await executor.execute(call)
        if result.error:
            raise RuntimeError(result.error_message)
        return result.output

    return _call


def _read_file_callable(executor: FileToolExecutor):
    async def _call(args: Mapping) -> object:
        from core.agent.types import ToolCall, ToolResult

        call = ToolCall(
            call_id="read_file_call",
            tool_name=READ_FILE_TOOL.name,
            arguments=dict(args),
        )
        result: ToolResult = await executor.execute(call)
        if result.error:
            raise RuntimeError(result.error_message)
        return result.output

    return _call


def _write_file_callable(executor: FileToolExecutor):
    async def _call(args: Mapping) -> object:
        from core.agent.types import ToolCall, ToolResult

        call = ToolCall(
            call_id="write_file_call",
            tool_name=WRITE_FILE_TOOL.name,
            arguments=dict(args),
        )
        result: ToolResult = await executor.execute(call)
        if result.error:
            raise RuntimeError(result.error_message)
        return result.output

    return _call


def _create_document_callable(executor: "DocumentToolExecutor"):
    from core.tools.document_tools import DocumentToolExecutor

    async def _call(args: Mapping) -> object:
        from core.agent.types import ToolCall, ToolResult

        call = ToolCall(
            call_id="create_document_call",
            tool_name="create_document",
            arguments=dict(args),
        )
        result: ToolResult = await executor.execute(call)
        if result.error:
            raise RuntimeError(result.error_message)
        return result.output

    return _call
