"""Framework-agnostic data types for the Agent Runtime.

These types are deliberately plain dataclasses — they must not depend on
FastAPI, Pydantic, or any other framework. The application layer
(``backend/app``) may wrap them in Pydantic DTOs for HTTP transport.

This mirrors the design convention of :mod:`core.llm.types` and
:mod:`core.routing.types`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Execution phases / state machine
# ---------------------------------------------------------------------------


class ExecutionPhase(str, Enum):
    """The current phase of the agent's state machine.

    The agent moves through these phases in order, but some phases (e.g.
    TOOL_EXECUTION → OBSERVATION → DECIDE_ACTION) can repeat within a single
    step.
    """

    START = "start"
    UNDERSTAND = "understand"
    PLAN = "plan"
    ROUTE_MODEL = "route_model"
    DECIDE_ACTION = "decide_action"
    TOOL_REQUEST = "tool_request"
    TOOL_EXECUTION = "tool_execution"
    OBSERVATION = "observation"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"


class AgentStatus(str, Enum):
    """High-level status of the agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanStep:
    """A single step within a plan.

    A step represents a logical unit of work (typically a tool call) that the
    agent intends to perform. The agent may deviate from the plan if
    observations require it.
    """

    step_id: str
    description: str
    tool_name: Optional[str] = None
    inputs: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class Plan:
    """A plan produced by the Planner.

    Plans are frozen to prevent accidental mutation during execution.
    """

    goal: str
    steps: tuple[PlanStep, ...] = field(default_factory=tuple)
    reasoning: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0


# ---------------------------------------------------------------------------
# Tool call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A single invocation of a tool.

    ``arguments`` is the raw argument dict as passed to the tool. It is NOT
    stored in a way that could expose sensitive data in logs.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    phase: ExecutionPhase = ExecutionPhase.TOOL_REQUEST
    step_index: int = 0


@dataclass(frozen=True)
class ToolResult:
    """The result of a tool execution.

    ``output`` is the tool's output data. ``error`` is set when the tool
    raised an exception; ``error_message`` is safe for display/logging.
    """

    call_id: str
    tool_name: str
    output: Any = None
    error: bool = False
    error_message: str = ""
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """A single observation made by the agent after a tool result.

    Observations are frozen strings that the agent uses to decide the next action.
    """

    call_id: str
    content: str
    source: str = ""  # tool_name or "model"


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentMessage:
    """A single message in the agent's conversation history."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str


# ---------------------------------------------------------------------------
# Artifact / source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """An artifact produced by the agent (e.g. a generated document, code)."""

    artifact_id: str
    artifact_type: str  # e.g. "text", "code", "summary"
    content: str
    source_tool: Optional[str] = None


@dataclass(frozen=True)
class Source:
    """A source referenced by the agent during reasoning."""

    source_id: str
    source_type: str  # e.g. "knowledge", "document"
    reference: str  # human-readable reference
    relevance: float = 1.0


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """The result of the verifier's check on the agent's final output."""

    passed: bool
    reason: str
    suggestions: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """The full mutable state of an agent run.

    This object is mutated by the agent as it executes. It is not frozen (the
    agent needs to append to messages, completed_steps, etc.). Access from
    outside the agent should be read-only; the public surface exports a frozen
    snapshot via :class:`AgentResult`.

    Attributes:
        task_id: Unique identifier for this run (e.g. UUID string).
        task: The original task string.
        status: High-level run status.
        current_phase: Current state-machine phase.
        messages: Conversation history.
        plan: The planner's current plan (updated as the agent executes).
        current_step: Index of the step the agent is working on.
        completed_steps: Number of steps completed so far.
        selected_model: The model selected by the router for this run.
        tool_calls: All tool calls made so far (in order).
        observations: All observations collected so far.
        artifacts: All artifacts produced.
        sources: All sources referenced.
        errors: Non-fatal errors encountered during execution.
        verification_result: Result of the final verification.
        iteration: Current iteration number (1-based).
        started_at: Monotonic timestamp when the run started (None if not started).
    """

    task_id: str
    task: str
    status: AgentStatus = AgentStatus.PENDING
    current_phase: ExecutionPhase = ExecutionPhase.START
    messages: list[AgentMessage] = field(default_factory=list)
    plan: Optional[Plan] = None
    current_step: int = 0
    completed_steps: int = 0
    selected_model: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    iteration: int = 0
    started_at: Optional[float] = None  # time.monotonic() on start


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for an agent run.

    These are the knobs that control the agent's behaviour and safety limits.
    All limits are enforced within the agent's execution loop.
    """

    max_iterations: int = 10
    """Maximum number of main loop iterations (plan/act/observe cycles)."""

    max_tool_calls: int = 30
    """Maximum number of tool calls allowed in a single run."""

    max_repetitive_tool_calls: int = 3
    """Maximum consecutive repetitions of the same tool with identical args.

    Exceeding this triggers :class:`RepetitiveToolCallError`.
    """

    execution_timeout_seconds: float = 120.0
    """Wall-clock timeout for the entire run."""

    allow_retries_on_tool_failure: bool = False
    """If True, the agent may retry a failed tool call once."""

    verbose: bool = False
    """If True, the agent emits verbose debug logs."""

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if self.max_repetitive_tool_calls < 1:
            raise ValueError("max_repetitive_tool_calls must be at least 1")
        if self.execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be positive")


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentResult:
    """A frozen snapshot of the agent's final state.

    Returned to the caller after the agent run finishes (success or failure).
    The underlying :class:`AgentState` is not frozen, but this result object
    is — it can be safely stored, logged, or returned over an HTTP boundary.
    """

    task_id: str
    task: str
    status: AgentStatus
    final_phase: ExecutionPhase
    messages: tuple[AgentMessage, ...]
    plan: Optional[Plan]
    completed_steps: int
    selected_model: Optional[str]
    tool_calls: tuple[ToolCall, ...]
    observations: tuple[Observation, ...]
    artifacts: tuple[Artifact, ...]
    sources: tuple[Source, ...]
    errors: tuple[str, ...]
    verification_result: Optional[VerificationResult]
    iteration: int
    error: Optional[str]  # short error type name for UI

    @classmethod
    def from_state(
        cls,
        state: AgentState,
        error: Optional[str] = None,
    ) -> "AgentResult":
        """Create a frozen result snapshot from the live state."""
        return cls(
            task_id=state.task_id,
            task=state.task,
            status=state.status,
            final_phase=state.current_phase,
            messages=tuple(state.messages),
            plan=state.plan,
            completed_steps=state.completed_steps,
            selected_model=state.selected_model,
            tool_calls=tuple(state.tool_calls),
            observations=tuple(state.observations),
            artifacts=tuple(state.artifacts),
            sources=tuple(state.sources),
            errors=tuple(state.errors),
            verification_result=state.verification_result,
            iteration=state.iteration,
            error=error,
        )
