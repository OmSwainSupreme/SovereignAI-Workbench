"""SovereignAI Workbench — Agent Runtime (Phase 4).

Public surface for the controlled, stateful, tool-using Agent Runtime.

The agent is a **selector** that uses the :class:`core.routing.ModelRouter` for
model selection and a :class:`core.llm.ModelGateway` for generation. It never
talks to Ollama or any provider directly.

Example::

    from core.agent import Agent, AgentConfig, ToolRegistry
    from core.routing import TaskType, RoutingRequest
    from core.agent.registry import DefaultToolRegistry
    from core.agent.executor import SyncToolExecutor
    from core.agent.planner import SimplePlanner
    from core.agent.verifier import SimpleVerifier

    registry = DefaultToolRegistry()
    executor = SyncToolExecutor(registry)
    planner = SimplePlanner()
    verifier = SimpleVerifier()

    agent = Agent(
        model_router=my_router,
        model_gateway=my_gateway,
        tool_executor=executor,
        planner=planner,
        verifier=verifier,
        max_iterations=10,
        max_tool_calls=20,
    )

    result = await agent.run("Explain the architecture of the solar system.")
    print(result.state.status, result.state.completed_steps)
"""
from core.agent.errors import (
    AgentExecutionError,
    AgentTimeoutError,
    IterationLimitExceededError,
    InvalidTaskError,
    MaxToolCallsExceededError,
    PlanningFailureError,
    RepetitiveToolCallError,
    RoutingFailureError,
    ToolExecutionError,
    UnknownToolError,
    VerificationFailureError,
)
from core.agent.agent import Agent
from core.agent.interfaces import (
    Planner,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    Verifier,
)
from core.agent.planner import SimplePlanner
from core.agent.types import (
    AgentConfig,
    AgentMessage,
    AgentResult,
    AgentState,
    AgentStatus,
    Artifact,
    ExecutionPhase,
    Observation,
    Plan,
    PlanStep,
    Source,
    ToolCall,
    ToolResult,
    VerificationResult,
)
from core.agent.verifier import SimpleVerifier
from core.agent.registry import SyncToolExecutor, DefaultToolRegistry

__all__ = [
    # types
    "AgentState",
    "AgentStatus",
    "AgentMessage",
    "Artifact",
    "ExecutionPhase",
    "Observation",
    "Plan",
    "PlanStep",
    "Source",
    "ToolCall",
    "ToolResult",
    "VerificationResult",
    "AgentConfig",
    "AgentResult",
    # interfaces
    "Planner",
    "ToolDefinition",
    "ToolExecutor",
    "ToolRegistry",
    "Verifier",
    # built-in implementations
    "SimplePlanner",
    "SimpleVerifier",
    "SyncToolExecutor",
    "DefaultToolRegistry",
    # errors
    "AgentExecutionError",
    "AgentTimeoutError",
    "InvalidTaskError",
    "IterationLimitExceededError",
    "MaxToolCallsExceededError",
    "PlanningFailureError",
    "RepetitiveToolCallError",
    "RoutingFailureError",
    "ToolExecutionError",
    "UnknownToolError",
    "VerificationFailureError",
    # agent
    "Agent",
]
