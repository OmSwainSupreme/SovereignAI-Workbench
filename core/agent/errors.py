"""Exception hierarchy for the Agent Runtime.

These errors are part of the public surface. The application layer should
catch :class:`AgentExecutionError` (the common base) and translate to a
domain-specific error or HTTP response as appropriate.

The hierarchy is deliberately small: each concrete error maps to a single,
diagnosable failure mode the caller can act on.
"""
from __future__ import annotations

from typing import Optional


class AgentExecutionError(Exception):
    """Base class for all agent errors.

    All other agent exceptions inherit from this. Callers can catch this to
    handle "the agent failed" generically, or catch a subclass for a more
    specific recovery.
    """


class InvalidTaskError(AgentExecutionError):
    """The supplied task is empty or otherwise unusable.

    Raised before any agent work is done. The agent never starts.
    """


class PlanningFailureError(AgentExecutionError):
    """The planner could not produce a plan.

    May carry the underlying planner's error in :attr:`cause`.
    """

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.cause = cause


class RoutingFailureError(AgentExecutionError):
    """The model router could not select a model for the task.

    Wraps the routing layer's :class:`core.routing.NoSuitableModelError` or
    other routing errors.
    """

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.cause = cause


class UnknownToolError(AgentExecutionError):
    """The agent requested a tool that is not registered.

    This is a programming error (the planner asked for a tool that does
    not exist), not a runtime user error.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Unknown tool: {tool_name!r}")


class ToolExecutionError(AgentExecutionError):
    """A tool was found but failed to execute.

    The underlying tool's error message is preserved on :attr:`cause` for
    diagnostics, but the agent does NOT include tool arguments in this
    exception to avoid leaking sensitive data.
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool {tool_name!r} execution failed: {message}")


class VerificationFailureError(AgentExecutionError):
    """The verifier rejected the agent's final output."""


class IterationLimitExceededError(AgentExecutionError):
    """The agent hit the configured maximum iteration limit.

    The state is preserved on the :class:`AgentResult` for diagnostics.
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent exceeded the maximum of {max_iterations} iterations"
        )


class MaxToolCallsExceededError(AgentExecutionError):
    """The agent hit the configured maximum number of tool calls."""

    def __init__(self, max_tool_calls: int) -> None:
        self.max_tool_calls = max_tool_calls
        super().__init__(
            f"Agent exceeded the maximum of {max_tool_calls} tool calls"
        )


class RepetitiveToolCallError(AgentExecutionError):
    """The agent requested the same tool with the same arguments too many times.

    This is a safety guard against loops the planner might get stuck in. The
    exact threshold is configurable on the agent.
    """


class AgentTimeoutError(AgentExecutionError):
    """The agent exceeded a wall-clock deadline."""
