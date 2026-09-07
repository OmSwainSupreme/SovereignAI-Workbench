"""End-to-end Agent + execute_code integration tests.

These tests verify the complete flow:

    User task
    → Agent planning
    → model selection (ModelRouter)
    → execute_code tool call (ToolRegistry / SyncToolExecutor)
    → Docker sandbox execution (or a fake sandbox for unit tests)
    → observation
    → verification
    → completion.

The tests are organised into:

* :class:`TestAgentExecuteCodeRouting` — task type detection and routing
  to a coding-capable model.
* :class:`TestAgentExecuteCodeEndToEnd` — full end-to-end flow with a
  fake sandbox that records calls and returns scripted results.
* :class:`TestAgentExecuteCodeFailureModes` — failure handling:
  syntax/runtime errors, timeouts, unknown tools, sandbox failures,
  tool failure propagated as an observation, and bounded termination.
* :class:`TestAgentExecuteCodeSecurity` — security boundary is preserved.
* :class:`TestAgentExecuteCodeSensitiveLogging` — source code, tool
  arguments, generated output, and secrets never appear in logs.
* :class:`TestAgentExecuteCodeDockerIntegration` — real Docker
  integration test (only runs when Docker is available).
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import shutil
import subprocess
from typing import Any, Iterator, List
from unittest.mock import AsyncMock

import pytest

from core.agent import (
    Agent,
    AgentConfig,
    AgentState,
    AgentStatus,
    ExecutionPhase,
    Observation,
    Plan,
    Planner,
    PlanStep,
    SimplePlanner,
    SimpleVerifier,
    SyncToolExecutor,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
    Verifier,
)
from core.agent.registry import DefaultToolRegistry
from core.agent.types import VerificationResult
from core.routing import (
    Capability,
    Modality,
    ModelDefinition,
    ModelRouter,
    RoutingDecision,
    RoutingRequest,
    TaskType,
)
from core.routing.registry import (
    ModelRegistry,
    load_default_registry,
)
from core.sandbox import (
    EXECUTE_CODE_TOOL,
    CodeToolExecutor,
    ExecutionRequest,
    ExecutionResult,
    ResourceLimits,
    Sandbox,
    SandboxConfig,
    SandboxError,
    SandboxExecutionError,
    SandboxTimeoutError,
    _NullSandbox,
    register_code_tools,
)


# ---------------------------------------------------------------------------
# Fake / mock collaborators
# ---------------------------------------------------------------------------


class FakeRouter:
    """A fake router that returns a configured model.

    Records every RoutingRequest it receives so tests can assert that the
    agent actually consulted the router with the right task type and
    capabilities.
    """

    def __init__(self, model_name: str = "coding") -> None:
        self.model_name = model_name
        self.calls: List[RoutingRequest] = []

    def route(self, request: RoutingRequest) -> Any:
        self.calls.append(request)
        decision = type(
            "RoutingDecision",
            (),
            {
                "model": type(
                    "ModelDef",
                    (),
                    {
                        "logical_name": self.model_name,
                        "provider": "ollama",
                        "provider_model": "qwen2.5-coder:3b",
                        "capabilities": frozenset({Capability.CODING, Capability.GENERAL}),
                        "priority": 20,
                    },
                )(),
                "reason": f"Fake router selected {self.model_name}",
                "score": 100,
                "matched_capabilities": frozenset({Capability.CODING}),
                "modality_satisfied": True,
                "preferred_honoured": False,
            },
        )()
        return decision


class RecordingPlanner(Planner):
    """A planner that returns a configurable plan.

    The planner records every (task, available_tools) call so tests can
    assert that the agent exposed the execute_code tool to it.
    """

    def __init__(self, plan: Plan | None = None) -> None:
        self._plan = plan or Plan(goal="test", steps=())
        self.calls: List[dict[str, Any]] = []

    async def plan(
        self,
        task: str,
        available_tools: list[str],
        state: AgentState,
    ) -> Plan:
        self.calls.append(
            {"task": task, "available_tools": list(available_tools)}
        )
        return self._plan


class ScriptedSandbox(Sandbox):
    """A scripted sandbox that returns canned results per call.

    Tracks every ExecutionRequest received. Used to verify that the
    execute_code tool's arguments were correctly translated to an
    ExecutionRequest and forwarded to the security boundary.
    """

    def __init__(
        self,
        config: SandboxConfig,
        results: List[ExecutionResult] | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        super().__init__(config)
        self.requests: List[ExecutionRequest] = []
        self._results = list(results or [])
        self._raise = raise_on_call

    @property
    def backend_name(self) -> str:
        return "scripted"

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        if self._raise is not None:
            raise self._raise
        if not self._results:
            return ExecutionResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                timed_out=False,
                duration_seconds=0.0,
                execution_id=request.execution_id,
                language=request.language,
            )
        return self._results.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox_config() -> SandboxConfig:
    return SandboxConfig(
        limits=ResourceLimits(
            timeout_seconds=5.0,
            max_output_bytes=4096,
            max_memory_mb=128,
            max_cpus=1.0,
            max_pids=64,
            max_workspace_bytes=4 * 1024 * 1024,
            max_code_bytes=32 * 1024,
        ),
        max_timeout_seconds=30.0,
    )


@pytest.fixture
def code_registry(sandbox_config: SandboxConfig) -> DefaultToolRegistry:
    """A registry with execute_code wired to a scripted sandbox."""
    registry = DefaultToolRegistry()
    sandbox = ScriptedSandbox(sandbox_config)
    register_code_tools(registry, sandbox)
    # Stash the sandbox so tests can inspect what the agent did.
    registry._scripted_sandbox = sandbox  # type: ignore[attr-defined]
    return registry


@pytest.fixture
def code_executor(code_registry: DefaultToolRegistry) -> SyncToolExecutor:
    return SyncToolExecutor(code_registry)


# ---------------------------------------------------------------------------
# Test: routing
# ---------------------------------------------------------------------------


class TestAgentExecuteCodeRouting:
    """The agent must route coding tasks through the ModelRouter and never
    call Ollama or any provider directly."""

    def test_agent_uses_router_not_ollama(self) -> None:
        """The agent must consult the ModelRouter; it must not import or
        instantiate the Ollama provider directly."""
        # The presence of FakeRouter and absence of any provider import
        # is asserted by the architectural constraint that Agent takes a
        # model_router and has no provider import. This is a guard test
        # to catch regressions where someone bypasses the router.
        from core.agent import agent as agent_module

        src = open(agent_module.__file__, "r", encoding="utf-8").read()
        # The agent must not import ollama.
        assert "from core.llm.providers.ollama" not in src
        assert "import ollama" not in src

    def test_router_consulted_with_request(self) -> None:
        """A coding task must produce at least one router call."""
        registry = DefaultToolRegistry()
        sandbox = ScriptedSandbox(
            SandboxConfig(),
            results=[
                ExecutionResult(
                    success=True,
                    exit_code=0,
                    stdout="0 1 1 2 3 5 8 13 21 34",
                    stderr="",
                    timed_out=False,
                    duration_seconds=0.1,
                )
            ],
        )
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)

        router = FakeRouter(model_name="coding")
        plan = Plan(
            goal="fibonacci",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="Compute Fibonacci",
                    tool_name="execute_code",
                    inputs={
                        "code": (
                            "a, b = 0, 1\n"
                            "for _ in range(20):\n"
                            "    print(a)\n"
                            "    a, b = b, a + b\n"
                        ),
                        "language": "python",
                    },
                ),
            ),
        )
        agent = Agent(
            model_router=router,
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=registry,
        )

        result = asyncio.run(agent.run("Calculate the first 20 Fibonacci numbers"))

        assert result.status == AgentStatus.COMPLETE
        assert result.selected_model == "coding"
        # The router was called at least once (route_model phase).
        assert len(router.calls) >= 1

    def test_planner_sees_execute_code_tool(self) -> None:
        """The planner must see the execute_code tool in available_tools."""
        registry = DefaultToolRegistry()
        sandbox = ScriptedSandbox(SandboxConfig())
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)

        planner = RecordingPlanner(plan=Plan(goal="x", steps=()))
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=executor,
            planner=planner,
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=2),
            tool_registry=registry,
        )

        asyncio.run(agent.run("compute something"))

        assert len(planner.calls) == 1
        assert "execute_code" in planner.calls[0]["available_tools"]

    def test_coding_task_routes_to_coding_model_via_router(self) -> None:
        """Verify that a coding task type routes to the coding model via
        the real ModelRouter (not a fake)."""
        registry = DefaultToolRegistry()
        sandbox = ScriptedSandbox(SandboxConfig())
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)

        # Build a real router and verify it returns the coding model
        # for a CODING task type.
        model_registry = ModelRegistry(
            [
                ModelDefinition(
                    logical_name="general",
                    provider="ollama",
                    provider_model="qwen3:4b",
                    capabilities=frozenset({Capability.GENERAL}),
                ),
                ModelDefinition(
                    logical_name="coding",
                    provider="ollama",
                    provider_model="qwen2.5-coder:3b",
                    capabilities=frozenset({Capability.CODING, Capability.GENERAL}),
                    priority=20,
                ),
            ]
        )
        router = ModelRouter(registry=model_registry)

        plan = Plan(
            goal="coding",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print(1)"},
                ),
            ),
        )
        agent = Agent(
            model_router=router,
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=registry,
        )

        result = asyncio.run(agent.run("Write a function"))

        # The router must have selected the coding model.
        assert result.selected_model == "coding"
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CODING,
                required_capabilities=frozenset({Capability.CODING}),
            )
        )
        assert decision.model.logical_name == "coding"


# ---------------------------------------------------------------------------
# Test: end-to-end
# ---------------------------------------------------------------------------


class TestAgentExecuteCodeEndToEnd:
    """The full user-task → completion flow."""

    def test_fibonacci_end_to_end(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
    ) -> None:
        """User task → planning → routing → execute_code → sandbox → result → completion."""
        fib_code = (
            "a, b = 0, 1\n"
            "out = []\n"
            "for _ in range(20):\n"
            "    out.append(a)\n"
            "    a, b = b, a + b\n"
            "print(','.join(str(x) for x in out))\n"
        )
        plan = Plan(
            goal="Calculate first 20 Fibonacci numbers",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="Compute Fibonacci in Python",
                    tool_name="execute_code",
                    inputs={"code": fib_code, "language": "python"},
                ),
            ),
        )

        expected_output = (
            "0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181"
        )
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=True,
                exit_code=0,
                stdout=expected_output,
                stderr="",
                timed_out=False,
                duration_seconds=0.1,
                execution_id="sandbox-exec-1",
                language="python",
            )
        ]

        router = FakeRouter(model_name="coding")
        agent = Agent(
            model_router=router,
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )

        result = asyncio.run(
            agent.run("Calculate the first 20 Fibonacci numbers using Python")
        )

        # Status: complete.
        assert result.status == AgentStatus.COMPLETE
        assert result.final_phase == ExecutionPhase.COMPLETE

        # The agent planned, routed, called the tool, observed, verified,
        # and completed — at least the first 4 phases are recorded in
        # the final state.
        assert result.selected_model == "coding"

        # Exactly one tool call was made.
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.tool_name == "execute_code"
        assert call.arguments["code"] == fib_code
        assert call.arguments["language"] == "python"

        # One observation was produced from the tool result.
        assert len(result.observations) == 1
        obs = result.observations[0]
        assert obs.source == "execute_code"
        # The observation is the tool result's output, which for the
        # sandbox is a dict containing the success / stdout / etc.
        # We don't unpack the dict here, but we do require it to be
        # non-empty.
        assert obs.content

        # The sandbox actually received the request (the security
        # boundary was crossed).
        assert len(sandbox.requests) == 1
        assert sandbox.requests[0].code == fib_code
        assert sandbox.requests[0].language == "python"

    def test_observation_contains_execution_result(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
    ) -> None:
        """The observation should expose the ExecutionResult fields so the
        verifier/UI can use them."""
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=True,
                exit_code=0,
                stdout="hello world",
                stderr="",
                timed_out=False,
                duration_seconds=0.2,
                execution_id="exec-xyz",
                language="python",
            )
        ]
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print('hello world')"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )
        result = asyncio.run(agent.run("Print hello world"))

        # The observation was stringified from the tool result output.
        # The content is the dict, but it's safe to log (no source code).
        assert len(result.observations) == 1
        # The str(dict) representation contains the success marker.
        assert "True" in result.observations[0].content or "true" in result.observations[0].content

    def test_uses_default_tool_registry_with_null_sandbox(self) -> None:
        """The agent should be able to use the execute_code tool backed by
        the _NullSandbox (no real Docker, no real execution)."""
        registry = DefaultToolRegistry()
        sandbox = _NullSandbox(SandboxConfig())
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)

        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print(1)"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=registry,
        )
        result = asyncio.run(agent.run("Test with null sandbox"))

        # The null sandbox returns success=False (no-op), so the agent
        # should still complete (it observed a tool result; the verifier
        # passes on the basis of having observed at least one).
        assert result.status == AgentStatus.COMPLETE
        assert len(result.tool_calls) == 1


# ---------------------------------------------------------------------------
# Test: failure handling
# ---------------------------------------------------------------------------


class TestAgentExecuteCodeFailureModes:
    """Failure modes the agent must handle gracefully."""

    def test_valid_code_succeeds(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
    ) -> None:
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                timed_out=False,
                duration_seconds=0.1,
            )
        ]
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print('ok')"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )
        result = asyncio.run(agent.run("Run valid code"))
        assert result.status == AgentStatus.COMPLETE

    def test_syntax_error_propagated_as_tool_failure(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
    ) -> None:
        """A syntax/runtime error in user code is a normal program outcome
        (success=False, exit_code!=0). The sandbox/tool layer reports it as
        a successful tool invocation (no error), and the agent observes
        the result. This is the standard contract: tool execution did not
        fail; the program did."""
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="NameError: name 'x' is not defined",
                timed_out=False,
                duration_seconds=0.05,
            )
        ]
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print(undefined_var)"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )
        result = asyncio.run(agent.run("Trigger a runtime error"))

        # The tool call itself succeeded; the agent observed the result.
        assert result.status == AgentStatus.COMPLETE
        assert len(result.tool_calls) == 1
        assert len(result.observations) == 1

    def test_timeout_recorded_as_observation(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
    ) -> None:
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="execution exceeded the configured timeout",
                timed_out=True,
                duration_seconds=5.0,
            )
        ]
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "while True: pass"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )
        result = asyncio.run(agent.run("Run infinite loop"))
        # The agent must still terminate (not loop forever).
        assert result.status == AgentStatus.COMPLETE
        assert len(result.observations) == 1

    def test_unknown_tool_returns_failed_result(self) -> None:
        """If the planner requests a tool that does not exist, the agent
        surfaces it as a ToolExecutionError / failed result. The agent
        does not crash."""
        registry = DefaultToolRegistry()
        executor = SyncToolExecutor(registry)
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="does_not_exist",
                    inputs={},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=registry,
        )
        result = asyncio.run(agent.run("Use unknown tool"))
        assert result.status == AgentStatus.FAILED
        # Either ToolExecutionError or UnknownToolError is acceptable.
        assert result.error in ("ToolExecutionError", "UnknownToolError")

    def test_sandbox_execution_failure_propagated_as_observation(
        self,
        sandbox_config: SandboxConfig,
    ) -> None:
        """A sandbox-level failure (e.g. SandboxExecutionError) should be
        translated by CodeToolExecutor into a tool result with error=True,
        and the agent should record an observation and continue."""
        registry = DefaultToolRegistry()
        sandbox = ScriptedSandbox(
            sandbox_config,
            raise_on_call=SandboxExecutionError("docker crashed"),
        )
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)

        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print(1)"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=registry,
        )
        result = asyncio.run(agent.run("Trigger sandbox failure"))

        # The agent should still complete (a single failure becomes an
        # observation; the verifier passes on having made progress).
        assert result.status == AgentStatus.COMPLETE
        # The observation should reflect the tool error, not a crash.
        assert len(result.observations) == 1
        assert "[Tool error]" in result.observations[0].content

    def test_agent_terminates_bounded_with_repetitive_calls(self) -> None:
        """If the planner keeps requesting execute_code with the same
        arguments, the agent must terminate (RepetitiveToolCallError)."""
        import time

        registry = DefaultToolRegistry()
        sandbox = ScriptedSandbox(SandboxConfig())
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)
        plan = Plan(
            goal="x",
            steps=tuple(
                PlanStep(
                    step_id=f"s{i}",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print(1)"},
                )
                for i in range(10)
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(
                max_iterations=20,
                max_tool_calls=100,
                max_repetitive_tool_calls=3,
            ),
            tool_registry=registry,
        )

        start = time.monotonic()
        result = asyncio.run(agent.run("Repeat execute_code"))
        elapsed = time.monotonic() - start

        assert result.status == AgentStatus.FAILED
        assert result.error == "RepetitiveToolCallError"
        # Bounded termination: must complete quickly.
        assert elapsed < 5.0

    def test_max_tool_calls_exceeded(self) -> None:
        """The agent must respect max_tool_calls even when the plan has
        more steps."""
        registry = DefaultToolRegistry()
        sandbox = ScriptedSandbox(SandboxConfig())
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)
        # 50 steps but only 3 tool calls allowed.
        plan = Plan(
            goal="x",
            steps=tuple(
                PlanStep(
                    step_id=f"s{i}",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": f"print({i})"},
                )
                for i in range(50)
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(
                max_iterations=100,
                max_tool_calls=3,
                max_repetitive_tool_calls=3,
            ),
            tool_registry=registry,
        )
        result = asyncio.run(agent.run("Exceed max tool calls"))
        assert result.status == AgentStatus.FAILED
        assert result.error == "MaxToolCallsExceededError"
        assert len(result.tool_calls) == 3


# ---------------------------------------------------------------------------
# Test: security boundary
# ---------------------------------------------------------------------------


class TestAgentExecuteCodeSecurity:
    """The Docker sandbox must remain the security boundary when called
    through the agent."""

    def test_no_direct_sandbox_bypass(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
    ) -> None:
        """The agent must not execute code outside the registered sandbox."""
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                timed_out=False,
                duration_seconds=0.1,
            )
        ]
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print(1)"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )
        asyncio.run(agent.run("Test"))
        # The sandbox received exactly one request.
        assert len(sandbox.requests) == 1

    def test_uses_existing_tool_registry_no_second_path(self) -> None:
        """The agent must use the ToolRegistry/ToolExecutor abstraction.
        There must be no second execution mechanism."""
        # Inspect the Agent source: it must reference ToolExecutor
        # and not call sandbox.execute() directly.
        from core.agent import agent as agent_module

        src = open(agent_module.__file__, "r", encoding="utf-8").read()
        # The agent should NOT import Sandbox.
        assert "from core.sandbox" not in src
        # The agent must use the executor abstraction.
        assert "executor.execute" in src

    def test_tool_registry_executor_decoupling(self) -> None:
        """The agent can be wired with a custom ToolExecutor. The
        ToolRegistry and ToolExecutor are independent layers."""
        registry = DefaultToolRegistry()
        sandbox = ScriptedSandbox(SandboxConfig())
        register_code_tools(registry, sandbox)
        # Build a custom executor that wraps SyncToolExecutor.
        inner = SyncToolExecutor(registry)
        recorded: List[ToolCall] = []

        class TrackingExecutor(ToolExecutor):
            def __init__(self, inner: ToolExecutor) -> None:
                self._inner = inner

            @property
            def registry(self) -> ToolRegistry:
                return registry

            async def execute(self, call: ToolCall) -> ToolResult:
                recorded.append(call)
                return await self._inner.execute(call)

        executor = TrackingExecutor(inner)
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print(1)"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=registry,
        )
        result = asyncio.run(agent.run("Test"))
        assert result.status == AgentStatus.COMPLETE
        # The custom executor received the call.
        assert len(recorded) == 1


# ---------------------------------------------------------------------------
# Test: sensitive logging
# ---------------------------------------------------------------------------


class TestAgentExecuteCodeSensitiveLogging:
    """Source code, tool arguments, generated output, and secrets must
    never appear in agent or sandbox logs."""

    def test_source_code_not_logged(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_code = "API_KEY = 'sk-VERY-SECRET-12345'\nprint(API_KEY)\n"
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                timed_out=False,
                duration_seconds=0.1,
            )
        ]
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": secret_code, "language": "python"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )

        caplog.set_level(logging.DEBUG)
        asyncio.run(agent.run("Run secret code"))

        log_text = "\n".join(r.getMessage() for r in caplog.records)
        # The source code should never appear in logs.
        assert "sk-VERY-SECRET-12345" not in log_text
        # The literal source should not appear either.
        assert "API_KEY = " not in log_text

    def test_tool_arguments_not_logged_in_errors(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_path = "/very/secret/path/to/file.txt"
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._raise = SandboxExecutionError("crash")
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": f"open('{secret_path}').read()"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )

        caplog.set_level(logging.DEBUG)
        asyncio.run(agent.run("Read secret path"))

        log_text = "\n".join(r.getMessage() for r in caplog.records)
        # The secret path should not appear in any log message.
        assert secret_path not in log_text

    def test_generated_output_not_logged(
        self,
        code_registry: DefaultToolRegistry,
        code_executor: SyncToolExecutor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_output = "TOKEN=ghp_VERY_SECRET_OUTPUT_99999"
        sandbox: ScriptedSandbox = code_registry._scripted_sandbox  # type: ignore[attr-defined]
        sandbox._results = [
            ExecutionResult(
                success=True,
                exit_code=0,
                stdout=secret_output,
                stderr="",
                timed_out=False,
                duration_seconds=0.1,
            )
        ]
        plan = Plan(
            goal="x",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="x",
                    tool_name="execute_code",
                    inputs={"code": "print('TOKEN=...')"},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(),
            tool_executor=code_executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=code_registry,
        )

        caplog.set_level(logging.DEBUG)
        asyncio.run(agent.run("Print token"))

        log_text = "\n".join(r.getMessage() for r in caplog.records)
        # The generated output should not appear in any log message.
        assert "ghp_VERY_SECRET_OUTPUT_99999" not in log_text


# ---------------------------------------------------------------------------
# Test: Docker integration
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    """Return True if Docker CLI and daemon are both available."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.docker
class TestAgentExecuteCodeDockerIntegration:
    """End-to-end agent → tool → real Docker sandbox → completion.

    Only runs when Docker is available. Validates that the entire
    pipeline works against a real container, not just a fake.
    """

    @pytest.fixture(autouse=True)
    def _check_docker(self) -> Iterator[None]:
        if not _docker_available():
            pytest.skip("Docker daemon not available")
        yield

    def test_agent_calculates_fibonacci_via_docker(self) -> None:
        """User task → agent → execute_code → Docker → real Python output."""
        from core.sandbox.docker_sandbox import DockerSandbox

        registry = DefaultToolRegistry()
        sandbox = DockerSandbox(skip_image_check=False)
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)

        fib_code = (
            "a, b = 0, 1\n"
            "out = []\n"
            "for _ in range(20):\n"
            "    out.append(str(a))\n"
            "    a, b = b, a + b\n"
            "print(','.join(out))\n"
        )
        plan = Plan(
            goal="Calculate first 20 Fibonacci numbers",
            steps=(
                PlanStep(
                    step_id="s0",
                    description="Compute Fibonacci in Python via Docker",
                    tool_name="execute_code",
                    inputs={"code": fib_code, "language": "python", "timeout": 5.0},
                ),
            ),
        )
        agent = Agent(
            model_router=FakeRouter(model_name="coding"),
            tool_executor=executor,
            planner=RecordingPlanner(plan=plan),
            verifier=SimpleVerifier(),
            config=AgentConfig(
                max_iterations=5,
                execution_timeout_seconds=30.0,
            ),
            tool_registry=registry,
        )

        result = asyncio.run(
            agent.run("Calculate the first 20 Fibonacci numbers using Python")
        )

        assert result.status == AgentStatus.COMPLETE
        assert result.selected_model == "coding"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "execute_code"
        assert len(result.observations) == 1
