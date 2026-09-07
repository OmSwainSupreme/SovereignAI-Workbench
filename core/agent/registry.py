"""Concrete implementations of the Agent Runtime's registries and executors.

This module provides:

* :class:`DefaultToolRegistry` — the default in-memory tool registry.
* :class:`SyncToolExecutor` — the default tool executor that runs a tool's
  registered callable.
* :func:`build_default_registry` — convenience builder for tests.

A tool is a :class:`ToolDefinition` plus a Python callable. The
:class:`SyncToolExecutor` is the bridge between the abstract executor
interface and Python callables.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Awaitable, Callable, Mapping, Union

from core.agent.errors import ToolExecutionError, UnknownToolError
from core.agent.interfaces import ToolDefinition, ToolExecutor, ToolRegistry
from core.agent.types import ToolCall, ToolResult


_logger = logging.getLogger("sovereign-ai.agent.tools")

#: A callable is either sync or async; both are accepted.
ToolCallable = Callable[[Mapping[str, Any]], Union[Any, Awaitable[Any]]]


# ---------------------------------------------------------------------------
# DefaultToolRegistry
# ---------------------------------------------------------------------------


class DefaultToolRegistry(ToolRegistry):
    """An in-memory :class:`ToolRegistry`.

    Tools are stored in a dict keyed by name. Re-registration replaces the
    prior entry (mirroring the ProviderRegistry convention from
    :mod:`core.llm.registry`).
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._callables: dict[str, ToolCallable] = {}

    def register(
        self,
        tool: ToolDefinition,
        callable: ToolCallable | None = None,
    ) -> None:
        """Register a :class:`ToolDefinition` and optionally its callable.

        The ``callable`` is what the executor will run. If a registry is only
        used to advertise tools (no execution), the callable may be omitted.
        """
        if not tool.name:
            raise ValueError("Tool name must be a non-empty string")
        self._tools[tool.name] = tool
        if callable is not None:
            self._callables[tool.name] = callable

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._callables.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name) from exc

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# SyncToolExecutor
# ---------------------------------------------------------------------------


class SyncToolExecutor(ToolExecutor):
    """A :class:`ToolExecutor` that runs registered Python callables.

    The executor is **agnostic** about whether the registered callable is
    sync or async — both are supported. Sync callables are invoked
    directly; async callables are awaited via :func:`asyncio.iscoroutine`.

    The executor never raises on tool failure: tool errors are caught and
    returned as :class:`ToolResult` with ``error=True``. This is the
    "principle of least surprise" for the agent — a tool that crashes
    should not crash the agent.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def execute(self, call: ToolCall) -> ToolResult:
        # Unknown tools → UnknownToolError. This is a programming error
        # (the planner referenced a tool that does not exist), and the
        # agent treats it as a non-recoverable error in the current step.
        if not self._registry.has(call.tool_name):
            raise UnknownToolError(call.tool_name)

        # Only DefaultToolRegistry supports callables; if a different
        # registry implementation is used we can only validate that the
        # tool exists (above) and return a "not executable" error.
        if not isinstance(self._registry, DefaultToolRegistry):
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message="Tool registry does not support execution",
                latency_ms=0.0,
            )

        callable_ = self._registry._callables.get(call.tool_name)  # noqa: SLF001
        if callable_ is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message="Tool has no callable registered",
                latency_ms=0.0,
            )

        start = time.monotonic()
        try:
            result = callable_(call.arguments)
            if inspect.isawaitable(result):
                result = await result
            latency_ms = (time.monotonic() - start) * 1000
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=result,
                error=False,
                error_message="",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            # Use type name only — do not include user-provided arguments
            # in the error message, since they may be sensitive.
            _logger.warning(
                "tool.fail  call_id=%s  tool=%s  latency_ms=%.1f  error_type=%s",
                call.call_id,
                call.tool_name,
                latency_ms,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message=f"{type(exc).__name__}: {str(exc)[:200]}",
                latency_ms=latency_ms,
            )


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_default_registry(
    tools: Mapping[str, tuple[ToolDefinition, ToolCallable]] | None = None,
) -> DefaultToolRegistry:
    """Build a default registry pre-populated with the given tools.

    Each entry is a ``(ToolDefinition, callable)`` pair.
    """
    registry = DefaultToolRegistry()
    if tools:
        for tool_def, callable_ in tools.values():
            registry.register(tool_def, callable_)
    return registry
