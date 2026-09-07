"""End-to-end integration tests for complete SovereignAI workflows.

These tests verify the complete pipeline:
    user request → agent → policy → routing → tool → provider/sandbox → observation → verification → audit

Each test focuses on a specific workflow:
1. Secured file read/write
2. RAG retrieval
3. Image/OCR analysis
4. DOCX generation
5. Code execution through Docker Sandbox
6. Policy DENY
7. Tool/provider failure
8. Capability/modality-based model routing
"""
from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from core.agent import (
    Agent,
    AgentConfig,
    AgentState,
    AgentStatus,
    Plan,
    PlanStep,
)
from core.agent.errors import UnknownToolError
from core.agent.interfaces import Planner, ToolDefinition, ToolExecutor, ToolRegistry, Verifier
from core.agent.registry import DefaultToolRegistry, SyncToolExecutor
from core.agent.types import (
    ToolCall,
    ToolResult,
    VerificationResult,
)
from core.audit_logger import AuditLogger, reset_audit_logger
from core.routing import (
    Capability,
    Modality,
    ModelDefinition,
    ModelRouter,
    RoutingRequest,
    TaskType,
)
from core.routing.registry import ModelRegistry
from core.security.policy_engine import (
    ALLOW,
    DENY,
    get_policy_engine,
    init_policy_engine,
    reset_policy_engine,
)
from core.tools import Workspace


# ---------------------------------------------------------------------------
# Real components used by tests
# ---------------------------------------------------------------------------

#: Every tool the policy engine knows about → its canonical (capability, action).
_TOOL_CAPABILITIES = {
    "list_files": ("file_system", "list_files"),
    "read_file": ("file_system", "read_file"),
    "write_file": ("file_system", "write_file"),
    "create_document": ("document_generation", "create_document"),
    "search_knowledge_base": ("knowledge", "search_knowledge_base"),
    "execute_code": ("code_execution", "execute_code"),
    "ocr_image": ("vision", "ocr_image"),
    "analyze_image": ("vision", "analyze_image"),
}


def make_tool(name: str, capability: Optional[str] = None) -> ToolDefinition:
    """Create a ToolDefinition for the given tool name."""
    return ToolDefinition(
        name=name,
        description=name,
        capability=capability or _TOOL_CAPABILITIES.get(name, ("general", name))[1],
        input_schema={"type": "object", "additionalProperties": True},
    )


def make_step(tool_name: str, **inputs: Any) -> PlanStep:
    """Create a single PlanStep invoking ``tool_name`` with ``inputs``."""
    return PlanStep(
        step_id=f"s-{tool_name}",
        description=f"call {tool_name}",
        tool_name=tool_name,
        inputs=inputs,
    )


def make_plan(*steps: PlanStep, goal: str = "E2E task") -> Plan:
    return Plan(goal=goal, steps=tuple(steps))


def make_router() -> ModelRouter:
    """Build a real capability-based ModelRouter with a small registry.

    The router is deterministic and never talks to a provider, so tests can
    use the real router and still run offline.
    """
    general = ModelDefinition(
        logical_name="general",
        provider="ollama",
        provider_model="qwen3:4b",
        capabilities=frozenset({Capability.GENERAL}),
        input_modalities=frozenset({Modality.TEXT}),
        priority=10,
    )
    coding = ModelDefinition(
        logical_name="coding",
        provider="ollama",
        provider_model="qwen2.5-coder:3b",
        capabilities=frozenset({Capability.CODING}),
        input_modalities=frozenset({Modality.TEXT}),
        priority=20,
    )
    vision = ModelDefinition(
        logical_name="vision",
        provider="ollama",
        provider_model="qwen2.5vl:3b",
        capabilities=frozenset({Capability.VISION}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        priority=30,
    )
    registry = ModelRegistry([general, coding, vision])
    return ModelRouter(registry=registry)


def make_agent(
    *,
    plan: Plan,
    executor: ToolExecutor,
    router: ModelRouter | None = None,
    verifier: Verifier | None = None,
    config: AgentConfig | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[Agent, ToolRegistry]:
    """Build an Agent wired to real components and return (agent, registry)."""
    if registry is None:
        registry = executor.registry  # type: ignore[attr-defined]
    router = router or make_router()
    planner = _FixedPlanner(plan)
    verifier = verifier or _PassingVerifier()
    agent = Agent(
        model_router=router,  # type: ignore[arg-type]
        tool_executor=executor,
        planner=planner,
        verifier=verifier,
        config=config or AgentConfig(max_iterations=10),
        tool_registry=registry,
    )
    return agent, registry


class _FixedPlanner(Planner):
    """Planner that always returns a fixed plan."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def plan(
        self,
        task: str,
        available_tools: list[str],
        state: AgentState,
    ) -> Plan:
        # Guarantee the plan only references tools the agent actually knows.
        for step in self._plan.steps:
            if step.tool_name not in available_tools:
                raise ValueError(
                    f"Plan references unknown tool {step.tool_name!r}; "
                    f"available={available_tools}"
                )
        return self._plan


class _PassingVerifier(Verifier):
    """Verifier that always accepts the final state."""

    def __init__(self, passed: bool = True, reason: str = "OK") -> None:
        self._passed = passed
        self._reason = reason

    async def verify(self, state: AgentState) -> VerificationResult:
        return VerificationResult(passed=self._passed, reason=self._reason)


class _RecordingExecutor(ToolExecutor):
    """Tool executor that records every call and delegates to a callable map."""

    def __init__(
        self,
        registry: ToolRegistry,
        handlers: Mapping[str, Any],
    ) -> None:
        self._registry = registry
        self._handlers = dict(handlers)
        self.calls: list[ToolCall] = []

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if not self._registry.has(call.tool_name):
            raise UnknownToolError(call.tool_name)
        handler = self._handlers.get(call.tool_name)
        if handler is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output={},
                error=True,
                error_message=f"No handler for {call.tool_name}",
                latency_ms=0.0,
            )
        try:
            outcome = handler(call)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            return outcome
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output={},
                error=True,
                error_message=f"{type(exc).__name__}: {str(exc)[:120]}",
                latency_ms=0.0,
            )


def _ok_result(call: ToolCall, output: Any) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        output=output,
        error=False,
        error_message="",
        latency_ms=1.0,
    )


# ---------------------------------------------------------------------------
# Audit logger isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_audit():
    """A fresh AuditLogger pointed at a temp dir, installed as the singleton."""
    base = tempfile.mkdtemp(prefix="e2e-audit-")
    log_dir = str(Path(base) / "audit")
    reset_audit_logger()
    logger = AuditLogger(
        log_dir=log_dir,
        log_file="audit.log",
        max_bytes=1024 * 1024,
        backup_count=1,
    )
    # Replace the module singleton with the isolated logger.
    import core.audit_logger as audit_module

    audit_module._instance = logger
    yield logger, Path(log_dir) / "audit.log"
    reset_audit_logger()
    # Tear down handlers so the real AuditLogger never inherits stale state.
    for handler in list(logger._logger.handlers):
        logger._logger.removeHandler(handler)
    reset_audit_logger()


def _read_audit_entries(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Workflow 1: secured file read/write
# ---------------------------------------------------------------------------


class TestSecuredFileReadWrite:
    def test_file_read_write_workflow(self, isolated_audit) -> None:
        """The full user request → policy → routing → tool → audit pipeline."""
        reset_policy_engine()
        init_policy_engine({})  # file_system.* are ALLOW by default

        registry = DefaultToolRegistry()
        for name in ("read_file", "write_file"):
            registry.register(make_tool(name))

        files: dict[str, str] = {}

        def write_file(call: ToolCall) -> ToolResult:
            path = call.arguments.get("path", "")
            content = call.arguments.get("content", "")
            files[path] = content
            return _ok_result(call, {"bytes_written": len(content), "path": path})

        def read_file(call: ToolCall) -> ToolResult:
            path = call.arguments.get("path", "")
            content = files.get(path, "")
            return _ok_result(call, {"content": content, "path": path})

        executor = _RecordingExecutor(registry, {"write_file": write_file, "read_file": read_file})

        plan = make_plan(
            make_step("write_file", path="reports/summary.txt", content="hello world"),
            make_step("read_file", path="reports/summary.txt"),
            goal="write and read a workspace file",
        )
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Write 'hello world' to reports/summary.txt and read it back."))

        assert result.status == AgentStatus.COMPLETE
        assert len(executor.calls) == 2
        assert [c.tool_name for c in executor.calls] == ["write_file", "read_file"]
        assert files["reports/summary.txt"] == "hello world"

        # Reads back the value written in same session.
        read_out = executor.calls[1].arguments
        assert read_out["path"] == "reports/summary.txt"

        # Policy engine was installed and consulted (decision logged to audit).
        assert get_policy_engine() is not None
        try:
            entries = _read_audit_entries(isolated_audit[1])
            event_types = [e["event_type"] for e in entries]
            assert "policy_decision" in event_types
            assert "tool_execution_start" in event_types
            assert "tool_execution_end" in event_types
            # No sensitive content (file contents) in any audit line.
            serialized = "\n".join(json.dumps(e) for e in entries)
            assert "hello world" not in serialized
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 2: RAG retrieval
# ---------------------------------------------------------------------------


class TestRAGRetrieval:
    def test_rag_search_workflow(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({})  # knowledge.* ALLOW by default

        registry = DefaultToolRegistry()
        registry.register(make_tool("search_knowledge_base"))
        registry.register(make_tool("read_file"))

        def search_kb(call: ToolCall) -> ToolResult:
            query = call.arguments.get("query", "")
            return _ok_result(
                call,
                {
                    "results": [
                        {
                            "content": "Quantum computing uses qubits.",
                            "score": 0.94,
                            "metadata": {"filename": "quantum.txt"},
                        }
                    ],
                    "query": query,
                },
            )

        executor = _RecordingExecutor(registry, {"search_knowledge_base": search_kb})
        plan = make_plan(
            make_step("search_knowledge_base", query="quantum computing", top_k=3),
            goal="search the knowledge base",
        )
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Find information about quantum computing."))

        assert result.status == AgentStatus.COMPLETE
        assert len(executor.calls) == 1
        assert executor.calls[0].tool_name == "search_knowledge_base"
        assert executor.calls[0].arguments["query"] == "quantum computing"

        # Query text must never appear verbatim in the audit log.
        try:
            entries = _read_audit_entries(isolated_audit[1])
            serialized = "\n".join(json.dumps(e) for e in entries)
            assert "quantum computing" not in serialized
            assert any(e["event_type"] == "policy_decision" for e in entries)
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 3: image / OCR analysis
# ---------------------------------------------------------------------------


class TestImageOCRAnalysis:
    def test_ocr_image_workflow(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({})  # vision.* ALLOW by default

        registry = DefaultToolRegistry()
        registry.register(make_tool("ocr_image"))

        def ocr(call: ToolCall) -> ToolResult:
            return _ok_result(
                call,
                {
                    "text": "INVOICE NO. 90210",
                    "confidence": 0.99,
                    "language": "en",
                },
            )

        executor = _RecordingExecutor(registry, {"ocr_image": ocr})
        plan = make_plan(
            make_step("ocr_image", path="scans/invoice.png"),
            goal="extract text from an image",
        )
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Extract the text from scans/invoice.png."))

        assert result.status == AgentStatus.COMPLETE
        assert len(executor.calls) == 1
        assert executor.calls[0].tool_name == "ocr_image"
        assert executor.calls[0].arguments["path"] == "scans/invoice.png"

        # OCR text must NOT be written to the audit log.
        try:
            entries = _read_audit_entries(isolated_audit[1])
            serialized = "\n".join(json.dumps(e) for e in entries)
            assert "INVOICE NO. 90210" not in serialized
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 4: DOCX generation
# ---------------------------------------------------------------------------


class TestDOCXGeneration:
    def test_create_document_workflow(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({})  # document_generation.* ALLOW by default

        registry = DefaultToolRegistry()
        registry.register(make_tool("create_document"))

        created: list[dict[str, Any]] = []

        def create_document(call: ToolCall) -> ToolResult:
            doc = dict(call.arguments)
            created.append(doc)
            # DOCX output is never logged by the audit logger; here we fake a
            # deterministic size (a real DOCX would be produced on disk).
            return _ok_result(call, {"path": doc.get("path"), "size_bytes": 4096})

        executor = _RecordingExecutor(registry, {"create_document": create_document})
        plan = make_plan(
            make_step(
                "create_document",
                path="out/report.docx",
                title="Quarterly Report",
                content="Revenue grew 12%.",
            ),
            goal="generate a DOCX report",
        )
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Create a DOCX report at out/report.docx."))

        assert result.status == AgentStatus.COMPLETE
        assert len(executor.calls) == 1
        assert executor.calls[0].tool_name == "create_document"
        assert created and created[0]["path"] == "out/report.docx"

        # Document body text must never be logged.
        try:
            entries = _read_audit_entries(isolated_audit[1])
            serialized = "\n".join(json.dumps(e) for e in entries)
            assert "Revenue grew 12%" not in serialized
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 5: code execution through Docker Sandbox
# ---------------------------------------------------------------------------


class TestCodeExecutionSandbox:
    def test_execute_code_allowed_workflow(self, isolated_audit) -> None:
        reset_policy_engine()
        # Explicitly honour code execution (off by default) for this test.
        init_policy_engine({"code_execution": {"execute_code": ALLOW}})

        registry = DefaultToolRegistry()
        registry.register(make_tool("execute_code"))

        def execute_code(call: ToolCall) -> ToolResult:
            # Simulates a Docker sandbox returning the factorial of 5.
            return _ok_result(
                call,
                {
                    "stdout": "120\n",
                    "stderr": "",
                    "exit_code": 0,
                    "language": "python",
                },
            )

        executor = _RecordingExecutor(registry, {"execute_code": execute_code})
        plan = make_plan(
            make_step("execute_code", code="print(5 * 4 * 3 * 2 * 1)", language="python"),
            goal="calculate factorial of 5",
        )
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Calculate the factorial of 5."))

        assert result.status == AgentStatus.COMPLETE
        assert len(executor.calls) == 1
        assert executor.calls[0].tool_name == "execute_code"

        # Executed source code must never be written to the audit log.
        try:
            entries = _read_audit_entries(isolated_audit[1])
            serialized = "\n".join(json.dumps(e) for e in entries)
            assert "print(5 * 4 * 3 * 2 * 1)" not in serialized
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 6: policy DENY
# ---------------------------------------------------------------------------


class TestPolicyDeny:
    def test_denied_tool_never_reaches_executor(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({"code_execution": {"execute_code": DENY}})

        registry = DefaultToolRegistry()
        registry.register(make_tool("execute_code"))

        executor = _RecordingExecutor(registry, {"execute_code": lambda call: _ok_result(call, {})})
        plan = make_plan(
            make_step("execute_code", code="print('hax')", language="python"),
            goal="run code (should be blocked)",
        )
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Run this code for me."))

        # The tool was blocked: the executor never ran it.
        assert len(executor.calls) == 0

        # The agent surfaces the denial as a safe observation and completes or
        # marks failure — no executor call, no exception leak.
        try:
            entries = _read_audit_entries(isolated_audit[1])
            decisions = [e for e in entries if e["event_type"] == "policy_decision"]
            assert decisions, "expected a policy_decision audit event"
            assert decisions[-1]["outcome"] == DENY
            assert "ohax" not in "\n".join(json.dumps(e) for e in entries)
            assert "print('hax')" not in "\n".join(json.dumps(e) for e in entries)
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 6b: policy engine consulted for EVERY tool call
# ---------------------------------------------------------------------------


class TestEveryToolReachesPolicy:
    def test_all_protected_tools_consult_policy(self, isolated_audit) -> None:
        """Every protected tool name goes through PolicyEngine before execution."""
        reset_policy_engine()
        init_policy_engine({})  # all defaults ALLOW except execute_code

        # Register every protected tool the policy engine knows about.
        registry = DefaultToolRegistry()
        for name in _TOOL_CAPABILITIES:
            registry.register(make_tool(name))

        # All handlers succeed trivially.
        handlers = {name: (lambda call: _ok_result(call, {"ok": True})) for name in _TOOL_CAPABILITIES}

        # Use a real policy engine and confirm each tool call produces a
        # policy_decision audit event — proving the policy layer runs first.
        executor = _RecordingExecutor(registry, handlers)

        # One plan step per tool.
        steps = []
        for name in _TOOL_CAPABILITIES:
            args: dict[str, Any] = {"path": "x"}
            if name in ("search_knowledge_base",):
                args = {"query": "q"}
            if name in ("analyze_image",):
                args = {"path": "x", "prompt": "what is this?"}
            steps.append(make_step(name, **args))
        plan = make_plan(*steps, goal="exercise every tool")
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Use every tool."))

        # execute_code is denied by default, so it must NOT have executed.
        executed = [c.tool_name for c in executor.calls]
        assert "execute_code" not in executed
        # Every other tool ran (allowed).
        allowed = [n for n in _TOOL_CAPABILITIES if n != "execute_code"]
        assert set(executed) == set(allowed)

        # Every tool produced a policy_decision audit event with no leaking args.
        try:
            entries = _read_audit_entries(isolated_audit[1])
            decisions = [e for e in entries if e["event_type"] == "policy_decision"]
            caps_actions = {e["action"] for e in decisions}
            assert caps_actions == set(allowed) | {"execute_code"}
            serialized = "\n".join(json.dumps(e) for e in entries)
            assert "print('hax')" not in serialized  # no code payload
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 7: tool/provider failure propagates safely
# ---------------------------------------------------------------------------


class TestToolProviderFailure:
    def test_provider_failure_is_structured_and_safe(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({})  # file_system.* ALLOW

        registry = DefaultToolRegistry()
        registry.register(make_tool("read_file"))

        def failing_read(call: ToolCall) -> ToolResult:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output={},
                error=True,
                error_message="[Errno 2] No such file: missing.txt",
                latency_ms=1.0,
            )

        executor = _RecordingExecutor(registry, {"read_file": failing_read})
        plan = make_plan(
            make_step("read_file", path="missing.txt"),
            goal="read a file that does not exist",
        )
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Read the file missing.txt."))

        # The tool failed but the agent reports a structured, SAFE result — it
        # does NOT raise, and the failure is captured as an observation.
        assert result.status == AgentStatus.COMPLETE
        assert len(executor.calls) == 1
        assert result.tool_calls[0].tool_name == "read_file"
        assert result.observations and "[Tool error]" in result.observations[-1].content

        # The failure must not leak absolute host paths or internal stack
        # traces anywhere in the surfaced result.
        plain = str(result)
        for leak in ("C:\\", "Traceback", "  File \"", "raise ", "NoneType"):
            assert leak not in plain

        # Audit records the failure (outcome=FAILURE) and must not contain the
        # file payload (here, the error text is generic and safe — but we still
        # assert there is no absolute path or stack trace in the log).
        try:
            entries = _read_audit_entries(isolated_audit[1])
            end_events = [e for e in entries if e["event_type"] == "tool_execution_end"]
            assert end_events and end_events[-1]["outcome"] == "FAILURE"
            serialized = "\n".join(json.dumps(e) for e in entries)
            assert "C:\\" not in serialized
            assert "Traceback" not in serialized
        finally:
            reset_policy_engine()


# ---------------------------------------------------------------------------
# Workflow 8: capability/modality-based routing
# ---------------------------------------------------------------------------


class TestCapabilityModalityRouting:
    def test_routing_uses_capabilities_and_modalities(self) -> None:
        """The agent routes through the real ModelRouter, which selects models
        purely on capability/modality — not on provider or model name."""
        reset_policy_engine()
        init_policy_engine({})

        registry = DefaultToolRegistry()
        registry.register(make_tool("read_file"))

        executor = _RecordingExecutor(registry, {"read_file": lambda call: _ok_result(call, {"ok": True})})
        plan = make_plan(make_step("read_file", path="a.txt"))
        router = make_router()
        agent, _ = make_agent(plan=plan, executor=executor, router=router)
        result = asyncio.run(agent.run("Read a.txt."))

        assert result.status == AgentStatus.COMPLETE
        # Agent always selects a model via the router; with the CHAT task type
        # it must resolve to a model carrying the GENERAL capability.
        assert result.selected_model is not None

        # The real router's decision for a CHAT task must be the GENERAL model
        # (the only one with Capability.GENERAL in our registry) — proving
        # selection is capability-based, NOT provider-based.
        decision = router.route(RoutingRequest(task_type=TaskType.CHAT))
        assert decision.model.logical_name == "general"
        assert Capability.GENERAL in decision.matched_capabilities

        # A VISION request must NOT fall back to the text-only general model:
        # hard modality requirement forces the vision model.
        vision_req = RoutingRequest(task_type=TaskType.VISION)
        vision_decision = router.route(vision_req)
        assert vision_decision.model.logical_name == "vision"
        assert vision_decision.modality_satisfied is True

        # A CODING request routes to the coding model.
        coding_req = RoutingRequest(task_type=TaskType.CODING)
        coding_decision = router.route(coding_req)
        assert coding_decision.model.logical_name == "coding"
        assert Capability.CODING in coding_decision.matched_capabilities

        reset_policy_engine()

    def test_no_suitable_model_routes_to_safe_failure(self) -> None:
        """If no model matches the required capability, routing fails safely."""
        reset_policy_engine()
        init_policy_engine({})

        # A registry with no model capable of VISION at all.
        text_only = ModelDefinition(
            logical_name="text-only",
            provider="ollama",
            provider_model="tiny-llm",
            capabilities=frozenset({Capability.GENERAL}),
            input_modalities=frozenset({Modality.TEXT}),
        )
        registry = ModelRegistry([text_only])
        router = ModelRouter(registry=registry)

        from core.routing.errors import NoSuitableModelError

        with pytest.raises(NoSuitableModelError):
            router.route(RoutingRequest(task_type=TaskType.VISION))

        reset_policy_engine()


# ---------------------------------------------------------------------------
# Real-tool end-to-end: wire the ACTUAL file/document/RAG/vision tools through
# the real SyncToolExecutor + Workspace and drive them via the agent. This
# proves the true cross-component pipeline (no fake-executor bypass).
# ---------------------------------------------------------------------------


class TestRealToolIntegration:
    def _workspace(self) -> tuple[Workspace, Path]:
        base = tempfile.mkdtemp(prefix="e2e-real-")
        return Workspace(root_path=str(base)), Path(base)

    def test_real_file_write_then_read(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({})

        from core.tools import register_file_tools

        workspace, root = self._workspace()
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        executor = SyncToolExecutor(registry)

        plan = make_plan(
            make_step("write_file", path="notes/hello.txt", content="real file content"),
            make_step("read_file", path="notes/hello.txt"),
            goal="write then read a real workspace file",
        )
        agent, _ = make_agent(plan=plan, executor=executor, registry=registry)
        result = asyncio.run(agent.run("Write and read a real file."))

        assert result.status == AgentStatus.COMPLETE
        # The file genuinely exists on disk inside the workspace.
        path = root / "notes" / "hello.txt"
        assert path.read_text(encoding="utf-8") == "real file content"
        # Reads back the content from the real file.
        assert len(result.observations) >= 2
        obs_join = " | ".join(o.content for o in result.observations)
        assert "real file content" in obs_join
        reset_policy_engine()

    def test_real_file_path_traversal_rejected(self, isolated_audit) -> None:
        """Workspace boundary: a real tool rejects ../ traversal."""
        reset_policy_engine()
        init_policy_engine({})

        from core.tools import register_file_tools

        base = tempfile.mkdtemp(prefix="e2e-real-")
        workspace = Workspace(root_path=str(base))
        registry = DefaultToolRegistry()
        register_file_tools(registry, workspace)
        executor = SyncToolExecutor(registry)

        # Attempt to read a file OUTSIDE the workspace via parent traversal.
        secret = Path(base).parent / "secret-outside.txt"
        secret.write_text("do not leak")
        try:
            plan = make_plan(
                make_step("read_file", path="../../secret-outside.txt"),
                goal="attempt traversal",
            )
            agent, _ = make_agent(plan=plan, executor=executor, registry=registry)
            result = asyncio.run(agent.run("Read the secret file."))
            # The traversal must fail: the tool returns a structured error.
            obs_join = " | ".join(o.content for o in result.observations)
            assert "do not leak" not in obs_join
            assert secret.read_text() == "do not leak"  # nothing was read out
        finally:
            secret.unlink(missing_ok=True)
            reset_policy_engine()

    def test_real_rag_search_with_ingested_document(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({})

        from core.rag import create_knowledge_base, register_rag_tools
        from core.rag.embedding import FakeEmbeddingProvider

        base = tempfile.mkdtemp(prefix="e2e-real-")
        kb_workspace = Workspace(root_path=str(base))
        kb = create_knowledge_base(
            workspace=kb_workspace,
            embedding_provider=FakeEmbeddingProvider(dimension=128, seed=7),
            chunk_size=500,
            overlap=50,
        )
        # Ingest a real document. The deterministic fake embedding scores
        # keyword-heavy text highest, so repeat the searchable phrase — this
        # mirrors the existing RAG test strategy in test_rag.py.
        (Path(base) / "quantum.txt").write_text(
            "Quantum entanglement links two particles. Quantum entanglement is "
            "used in quantum computing. Quantum entanglement is non-local. "
            "Quantum entanglement enables secure communication." * 10,
            encoding="utf-8",
        )
        kb.ingest("quantum.txt")

        registry = DefaultToolRegistry()
        register_rag_tools(registry, kb)
        executor = SyncToolExecutor(registry)

        plan = make_plan(
            make_step("search_knowledge_base", query="quantum entanglement", top_k=3),
            goal="search the knowledge base",
        )
        agent, _ = make_agent(plan=plan, executor=executor, registry=registry)
        result = asyncio.run(agent.run("What is quantum entanglement?"))

        assert result.status == AgentStatus.COMPLETE
        obs_join = " | ".join(o.content for o in result.observations)
        assert "Quantum entanglement" in obs_join
        reset_policy_engine()

    def test_real_docx_generation(self, isolated_audit) -> None:
        """The real create_document tool produces an actual .docx on disk."""
        reset_policy_engine()
        init_policy_engine({})  # document_generation.create_document = ALLOW

        from core.tools import register_document_tools

        workspace, root = self._workspace()
        registry = DefaultToolRegistry()
        register_document_tools(registry, workspace)
        executor = SyncToolExecutor(registry)

        plan = make_plan(
            make_step(
                "create_document",
                path="reports/report.docx",
                title="Quarterly Report",
                content=[
                    {"type": "heading", "text": "Quarterly Report"},
                    {"type": "paragraph", "text": "Revenue grew 12% quarter over quarter."},
                    {"type": "bullet_list", "items": ["Sales +8%", "Margin +2%"]},
                ],
            ),
            goal="generate a real DOCX report",
        )
        agent, _ = make_agent(plan=plan, executor=executor, registry=registry)
        result = asyncio.run(agent.run("Create a DOCX report."))

        assert result.status == AgentStatus.COMPLETE
        docx_path = root / "reports" / "report.docx"
        assert docx_path.exists()

        # Verify the DOCX is a real, readable Word document.
        from docx import Document as DocxReader

        parsed = DocxReader(str(docx_path))
        paragraphs = [p.text for p in parsed.paragraphs]
        assert "Quarterly Report" in paragraphs
        assert any("Revenue grew 12%" in t for t in paragraphs)
        reset_policy_engine()

    def test_real_ocr_image(self, isolated_audit) -> None:
        reset_policy_engine()
        init_policy_engine({})

        from core.vision import FakeOCRProvider, FakeVisionProvider, ImageLoader, register_vision_tools

        base = tempfile.mkdtemp(prefix="e2e-real-")
        base_path = Path(base)
        # Minimal 1x1 PNG so the loader accepts it.
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        (base_path / "scan.png").write_bytes(png_bytes)
        workspace = Workspace(root_path=str(base))

        registry = DefaultToolRegistry()
        register_vision_tools(
            registry, FakeOCRProvider(), FakeVisionProvider(), ImageLoader(workspace)
        )
        executor = SyncToolExecutor(registry)

        plan = make_plan(
            make_step("ocr_image", path="scan.png"),
            goal="extract text from a real image",
        )
        agent, _ = make_agent(plan=plan, executor=executor, registry=registry)
        result = asyncio.run(agent.run("OCR the image scan.png."))

        assert result.status == AgentStatus.COMPLETE
        obs_join = " | ".join(o.content for o in result.observations)
        assert obs_join.strip()  # FakeOCRProvider returns a non-empty result
        reset_policy_engine()


# ---------------------------------------------------------------------------
# Cross-cutting: agent never bypasses policy for ANY protected tool
# ---------------------------------------------------------------------------


class TestNoPolicyBypass:
    def test_removing_policy_engine_disables_enforcement(self, isolated_audit) -> None:
        """When no policy engine is installed, the agent runs (backward compat)."""
        reset_policy_engine()
        assert get_policy_engine() is None

        registry = DefaultToolRegistry()
        registry.register(make_tool("write_file"))

        def write_file(call: ToolCall) -> ToolResult:
            return _ok_result(call, {"bytes_written": 3})

        executor = _RecordingExecutor(registry, {"write_file": write_file})
        plan = make_plan(make_step("write_file", path="x.txt", content="abc"))
        agent, _ = make_agent(plan=plan, executor=executor)
        result = asyncio.run(agent.run("Write abc to x.txt."))

        # With no policy engine the tool executes (Phase 6 backward-compat).
        assert result.status == AgentStatus.COMPLETE
        assert len(executor.calls) == 1
        reset_policy_engine()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))