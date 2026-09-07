"""Abstract interfaces for the Agent Runtime.

These interfaces are deliberately small and composable. The agent does not
know about concrete implementations — only the contracts.

The agent requires four collaborators:

* :class:`Planner` — produces a :class:`Plan` for a task.
* :class:`ToolRegistry` — knows about available :class:`ToolDefinition`s.
* :class:`ToolExecutor` — runs a tool and returns a :class:`ToolResult`.
* :class:`Verifier` — checks the agent's final output.

Each of these is implemented as a separate, swappable component. Tests
inject fakes; production may inject real implementations later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence

from core.agent.types import (
    AgentMessage,
    AgentState,
    Plan,
    PlanStep,
    ToolCall,
    ToolResult,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner(ABC):
    """Produces a :class:`Plan` for a task.

    A planner inspects the task, the available tools, and any prior
    conversation messages, and returns a :class:`Plan`. The planner does not
    have to be deterministic; it can be as simple as returning a single
    generic step, or as elaborate as a multi-step LLM-driven plan.

    Implementations MUST:

    * Return a :class:`PlanningFailureError`-raising plan (or raise directly)
      if they cannot produce a plan.
    * Never return a plan that references a tool name not in the
      ``available_tools`` argument.
    """

    @abstractmethod
    async def plan(
        self,
        task: str,
        available_tools: Sequence[str],
        state: AgentState,
    ) -> Plan:
        """Produce a plan for the given task.

        Args:
            task: The user's task string.
            available_tools: Names of tools currently available in the
                tool registry. Planners may use this to choose what tools
                to plan with.
            state: The current agent state (read-only). Useful for planners
                that need to inspect prior messages or observations.

        Returns:
            A :class:`Plan` object. May be empty for tasks that need no
            tool calls (e.g. simple Q&A).

        Raises:
            Exception: Implementations may raise. The agent will catch and
                surface the error as :class:`PlanningFailureError`.
        """


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class ToolDefinition:
    """A declared, registered tool.

    A tool is a named, callable unit of work that the agent may invoke. The
    tool itself is implemented elsewhere; this object is just its declaration.

    The :class:`ToolDefinition` is intentionally schema-light: ``input_schema``
    is a free-form mapping that callers (and the planner) can interpret
    however they like. We do not yet require a JSON-schema-like validator
    for inputs — that is the responsibility of the tool's executor.
    """

    __slots__ = (
        "name",
        "description",
        "input_schema",
        "output_description",
        "capability",
    )

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Optional[Mapping[str, Any]] = None,
        output_description: str = "",
        capability: str = "general",
    ) -> None:
        if not name:
            raise ValueError("Tool name must be a non-empty string")
        self.name = name
        self.description = description
        self.input_schema = dict(input_schema) if input_schema is not None else {}
        self.output_description = output_description
        self.capability = capability

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"ToolDefinition(name={self.name!r}, capability={self.capability!r})"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry(ABC):
    """An abstract registry of available :class:`ToolDefinition`s.

    The agent uses the registry to (a) enumerate available tools for the
    planner, and (b) reject unknown tool names.
    """

    @abstractmethod
    def register(self, tool: ToolDefinition) -> None:
        """Add a tool to the registry. Re-registration replaces prior entry."""

    @abstractmethod
    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if absent."""

    @abstractmethod
    def has(self, name: str) -> bool:
        """Return True if a tool is registered under this name."""

    @abstractmethod
    def get(self, name: str) -> ToolDefinition:
        """Return the tool by name. Raises :class:`UnknownToolError` if absent."""

    @abstractmethod
    def names(self) -> list[str]:
        """Return all registered tool names in insertion order."""


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class ToolExecutor(ABC):
    """An abstract executor that runs a tool call.

    The executor is what the agent calls when it has a :class:`ToolCall`
    ready. The executor is responsible for:

    * Looking up the tool (via the registry) and raising
      :class:`UnknownToolError` if not found.
    * Running the tool.
    * Catching exceptions and translating to :class:`ToolResult` with
      ``error=True`` (so the agent can observe and decide what to do).
    * Setting ``error_message`` to a safe, non-sensitive string.
    """

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return its result.

        Implementations MUST NOT raise on tool failure — they should return a
        :class:`ToolResult` with ``error=True`` and a safe ``error_message``.
        Implementations may raise on infrastructure failures (e.g. the
        registry cannot be consulted).
        """


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class Verifier(ABC):
    """An abstract verifier that checks the agent's final output.

    The verifier is called once the agent decides it is done. It returns a
    :class:`VerificationResult` indicating whether the work is acceptable.
    """

    @abstractmethod
    async def verify(self, state: AgentState) -> VerificationResult:
        """Inspect the agent's final state and return a verdict.

        Args:
            state: A read-only view of the agent's final state.

        Returns:
            A :class:`VerificationResult` with ``passed=True/False``.
        """
