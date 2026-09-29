"""Integration tests for Policy Engine with Agent Runtime."""

from __future__ import annotations

from typing import Any

import pytest

from core.agent import (
    Agent,
    AgentConfig,
    Plan,
    PlanStep,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
)
from core.agent.errors import UnknownToolError
from core.agent.registry import DefaultToolRegistry
from core.agent.types import AgentState, VerificationResult
from core.routing import Capability
from core.agent.interfaces import ToolRegistry
from core.security.policy_engine import (
    get_policy_engine,
    init_policy_engine,
    reset_policy_engine,
)


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


class FakeToolExecutor(ToolExecutor):
    """Tool executor that records calls and returns fake results."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self.calls: list[Any] = []

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def execute(self, call: Any) -> ToolResult:
        self.calls.append(call)
        if not self._registry.has(call.tool_name):
            raise UnknownToolError(call.tool_name)
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output={"result": f"fake result for {call.tool_name}"},
            error=False,
            error_message="",
            latency_ms=1.0,
        )


class FakePlanner:
    """A planner that produces a configurable plan."""

    def __init__(self, plan: Plan | None = None) -> None:
        self._plan = plan or Plan(goal="test", steps=())

    async def plan(
        self,
        task: str,
        available_tools: list[str],
        state: AgentState,
    ) -> Plan:
        return self._plan


class FakeVerifier:
    """A verifier that always passes."""

    async def verify(self, state: AgentState) -> VerificationResult:
        return VerificationResult(passed=True, reason="OK")


class FakeRouter:
    """A fake router that returns a configured model."""

    def __init__(self, model_name: str = "general") -> None:
        self.model_name = model_name

    def route(self, request: Any) -> Any:
        return type(
            "RoutingDecision",
            (),
            {
                "model": type(
                    "ModelDef",
                    (),
                    {
                        "logical_name": self.model_name,
                        "provider": "ollama",
                        "provider_model": "qwen3:4b",
                        "capabilities": frozenset({Capability.GENERAL}),
                        "priority": 10,
                    },
                )(),
                "reason": f"Fake router selected {self.model_name}",
                "score": 100,
                "matched_capabilities": frozenset({Capability.GENERAL}),
                "modality_satisfied": True,
                "preferred_honoured": False,
            },
        )()


class _FakeModelGateway:
    """A fake model gateway that returns a fixed response.

    Used to satisfy the :class:`Agent` constructor's required ``model_gateway``.
    """

    async def generate(self, request) -> Any:
        return type(
            "GenerationResponse",
            (),
            {
                "content": "fake response",
                "model": request.model or "fake-model",
            },
        )()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_agent(
    tool_name: str,
    capability: str = "general",
    plan: Plan | None = None,
) -> tuple[Agent, FakeToolExecutor]:
    """Build an Agent with a single registered tool and a given plan."""
    tool_registry = DefaultToolRegistry()
    tool_registry.register(
        ToolDefinition(
            name=tool_name,
            description=f"Fake {tool_name}",
            input_schema={},
            output_description="result",
            capability=capability,
        )
    )
    executor = FakeToolExecutor(tool_registry)
    planner = FakePlanner(plan=plan)
    agent = Agent(
        model_router=FakeRouter(),
        model_gateway=_FakeModelGateway(),
        tool_executor=executor,
        planner=planner,
        verifier=FakeVerifier(),
        config=AgentConfig(max_iterations=5),
        tool_registry=tool_registry,
    )
    return agent, executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPolicyIntegration:
    """Test Policy Engine integration with Agent Runtime."""

    @pytest.fixture(autouse=True)
    def _cleanup_global_engine(self) -> None:
        """Ensure the global policy engine is reset after every test."""
        yield
        reset_policy_engine()

    # --- ALLOW ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_allowed_tool_executes_normally(self) -> None:
        """A tool that is ALLOW in the default policy should execute."""
        init_policy_engine()  # default config: read_file → ALLOW

        plan = Plan(
            goal="read file",
            steps=(PlanStep(
                step_id="s0",
                description="Read",
                tool_name="read_file",
                inputs={"path": "test.txt"},
            ),),
        )
        agent, executor = _build_agent("read_file", capability="file_system", plan=plan)

        result = await agent.run("Read a file")

        assert result.status == "complete"
        assert len(executor.calls) == 1
        assert executor.calls[0].tool_name == "read_file"

    # --- DENY ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_denied_tool_is_blocked(self) -> None:
        """A tool that is DENY should not reach the executor."""
        init_policy_engine(config={
            "file_system": {"read_file": "DENY"},
            "default": "DENY",
        })

        plan = Plan(
            goal="read file",
            steps=(PlanStep(
                step_id="s0",
                description="Read",
                tool_name="read_file",
                inputs={"path": "test.txt"},
            ),),
        )
        agent, executor = _build_agent("read_file", capability="file_system", plan=plan)

        result = await agent.run("Read a file")

        assert result.status == "complete"
        assert len(executor.calls) == 0  # tool never reached executor
        assert any("[Policy Denied]" in o.content for o in result.observations)

    # --- REQUIRE_APPROVAL ----------------------------------------------

    @pytest.mark.asyncio
    async def test_approval_required_tool_is_blocked(self) -> None:
        """A tool that is REQUIRE_APPROVAL is treated as blocked (no approval workflow yet)."""
        init_policy_engine(config={
            "file_system": {"write_file": "REQUIRE_APPROVAL"},
            "default": "DENY",
        })

        plan = Plan(
            goal="write file",
            steps=(PlanStep(
                step_id="s0",
                description="Write",
                tool_name="write_file",
                inputs={"path": "out.txt", "content": "hi"},
            ),),
        )
        agent, executor = _build_agent("write_file", capability="file_system", plan=plan)

        result = await agent.run("Write a file")

        assert result.status == "complete"
        assert len(executor.calls) == 0
        assert any("[Approval Required]" in o.content for o in result.observations)

    # --- unknown tool → default DENY -----------------------------------

    @pytest.mark.asyncio
    async def test_unknown_tool_denied_by_default(self) -> None:
        """An unrecognised tool name should hit the default-DENY policy."""
        init_policy_engine()

        plan = Plan(
            goal="unknown",
            steps=(PlanStep(
                step_id="s0",
                description="Do thing",
                tool_name="totally_unknown_tool_xyz",
                inputs={},
            ),),
        )
        agent, executor = _build_agent("totally_unknown_tool_xyz", capability="general", plan=plan)

        result = await agent.run("Do something")

        assert result.status == "complete"
        assert len(executor.calls) == 0
        assert any("[Policy Denied]" in o.content for o in result.observations)

    # --- no policy engine → bypass (backward compat) -------------------

    @pytest.mark.asyncio
    async def test_no_policy_engine_allows_everything(self) -> None:
        """When no policy engine is set, all tools execute (backward compat)."""
        # Ensure global is None
        reset_policy_engine()

        plan = Plan(
            goal="read file",
            steps=(PlanStep(
                step_id="s0",
                description="Read",
                tool_name="read_file",
                inputs={"path": "x.txt"},
            ),),
        )
        agent, executor = _build_agent("read_file", capability="file_system", plan=plan)

        result = await agent.run("Read a file")

        assert result.status == "complete"
        assert len(executor.calls) == 1  # executed without policy check

    # --- multiple steps mixed allow/deny -------------------------------

    @pytest.mark.asyncio
    async def test_mixed_steps_one_denied_one_allowed(self) -> None:
        """In a multi-step plan, a denied step is skipped while allowed steps execute."""
        init_policy_engine(config={
            "file_system": {"read_file": "ALLOW", "write_file": "DENY"},
            "default": "DENY",
        })

        plan = Plan(
            goal="read then write",
            steps=(
                PlanStep(step_id="s0", description="Read", tool_name="read_file", inputs={"path": "a.txt"}),
                PlanStep(step_id="s1", description="Write", tool_name="write_file", inputs={"path": "b.txt", "content": "x"}),
            ),
        )

        tool_registry = DefaultToolRegistry()
        tool_registry.register(ToolDefinition(name="read_file", description="read", capability="file_system"))
        tool_registry.register(ToolDefinition(name="write_file", description="write", capability="file_system"))
        executor = FakeToolExecutor(tool_registry)

        agent = Agent(
            model_router=FakeRouter(),
            model_gateway=_FakeModelGateway(),
            tool_executor=executor,
            planner=FakePlanner(plan=plan),
            verifier=FakeVerifier(),
            config=AgentConfig(max_iterations=5),
            tool_registry=tool_registry,
        )

        result = await agent.run("Read then write")

        assert result.status == "complete"
        # Only read_file should have executed; write_file was denied
        assert len(executor.calls) == 1
        assert executor.calls[0].tool_name == "read_file"
        assert any("[Policy Denied]" in o.content for o in result.observations)

    # --- runtime policy update -----------------------------------------

    @pytest.mark.asyncio
    async def test_runtime_policy_update_takes_effect(self) -> None:
        """update_policy should immediately affect subsequent evaluations."""
        engine = init_policy_engine()

        plan = Plan(
            goal="read",
            steps=(PlanStep(step_id="s0", description="Read", tool_name="read_file", inputs={"path": "a.txt"}),),
        )
        agent, executor = _build_agent("read_file", capability="file_system", plan=plan)

        # Initially allowed
        result = await agent.run("Read a file")
        assert len(executor.calls) == 1

        # Now deny at runtime
        engine.update_policy("file_system", "read_file", "DENY")
        executor.calls.clear()
        result = await agent.run("Read again")
        assert len(executor.calls) == 0
        assert any("[Policy Denied]" in o.content for o in result.observations)

    # --- structured denial result (no sensitive data) ------------------

    @pytest.mark.asyncio
    async def test_deny_observation_exposes_no_sensitive_data(self) -> None:
        """The denial observation must not leak paths, prompts, or internals."""
        init_policy_engine(config={
            "file_system": {"read_file": "DENY"},
            "default": "DENY",
        })

        plan = Plan(
            goal="secret read",
            steps=(PlanStep(step_id="s0", description="Read secret", tool_name="read_file", inputs={"path": "/etc/shadow"}),),
        )
        agent, executor = _build_agent("read_file", capability="file_system", plan=plan)

        result = await agent.run("Read /etc/shadow")

        for obs in result.observations:
            if "[Policy Denied]" in obs.content:
                assert "/etc/shadow" not in obs.content
                assert "secret" not in obs.content.lower()
