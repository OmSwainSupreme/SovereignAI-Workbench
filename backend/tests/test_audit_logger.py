"""Tests for the centralized, secure local Audit Logger (Phase 7).

Covers event structure, privacy/redaction, policy DENY logging, successful and
failed execution logging, correlation IDs, malformed events, and logger failure
isolation.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest

from core.audit_logger import AuditLogger, get_audit_logger, reset_audit_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_entries(log_path: Path) -> list[dict[str, Any]]:
    """Read all JSON-lines from an audit log file."""
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


@pytest.fixture
def audit_dir():
    """Provide a temp audit directory and a fresh logger pointed at it."""
    base = tempfile.mkdtemp(prefix="audit-test-")
    log_dir = Path(base) / "audit"
    log_path = log_dir / "audit.log"
    logger = AuditLogger(
        log_dir=str(log_dir),
        log_file="audit.log",
        max_bytes=1024 * 1024,
        backup_count=2,
    )
    yield {"logger": logger, "log_dir": log_dir, "log_path": log_path}
    logger._logger.handlers.clear()


# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------


class TestEventStructure:
    def test_every_event_has_required_fields(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger.log_policy_decision(
            capability="file_system",
            action="read_file",
            decision="DENY",
            reason="Denied by policy",
            correlation_id="corr-123",
        )
        logger.log_tool_execution_start(
            tool_name="read_file",
            correlation_id="corr-123",
        )
        entries = _read_entries(audit_dir["log_path"])
        assert len(entries) >= 2
        for entry in entries:
            assert "timestamp" in entry
            assert "event_type" in entry
            assert "outcome" in entry

    def test_policy_decision_event_shape(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger.log_policy_decision(
            capability="code_execution",
            action="execute_code",
            decision="DENY",
            reason="Denied by policy: code_execution.execute_code",
            correlation_id="corr-1",
        )
        entry = _read_entries(audit_dir["log_path"])[-1]
        assert entry["event_type"] == "policy_decision"
        assert entry["capability"] == "code_execution"
        assert entry["action"] == "execute_code"
        assert entry["outcome"] == "DENY"
        assert entry["correlation_id"] == "corr-1"
        assert entry["details"]["reason"] == (
            "Denied by policy: code_execution.execute_code"
        )

    def test_tool_execution_end_outcome_success_failure(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger.log_tool_execution_end(
            tool_name="read_file",
            correlation_id="c1",
            success=True,
            latency_ms=1.5,
        )
        logger.log_tool_execution_end(
            tool_name="read_file",
            correlation_id="c2",
            success=False,
            error="File not found",
        )
        entries = _read_entries(audit_dir["log_path"])
        end_entries = [e for e in entries if e["event_type"] == "tool_execution_end"]
        assert end_entries[0]["outcome"] == "SUCCESS"
        assert end_entries[0]["details"]["latency_ms"] == 1.5
        assert end_entries[1]["outcome"] == "FAILURE"
        assert end_entries[1]["details"]["error"] == "File not found"


# ---------------------------------------------------------------------------
# Privacy / redaction
# ---------------------------------------------------------------------------


class TestPrivacyRedaction:
    def test_sensitive_details_keys_are_redacted(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger.log_tool_execution_start(
            tool_name="write_file",
            correlation_id="c1",
        )
        # Directly exercise the sanitizer via a public log path that passes
        # details, to confirm sensitive keys never reach the file.
        logger._info(
            event_type="tool_execution_end",
            capability="write_file",
            action="end",
            outcome="SUCCESS",
            correlation_id="c1",
            details={
                "path": "notes.txt",
                "content": "the quick brown fox secret",
                "code": "print('hi')",
                "prompt": "write a poem",
                "token": "abc123",
                "ok": "safe-value",
            },
        )
        entry = _read_entries(audit_dir["log_path"])[-1]
        details = entry["details"]
        assert details["content"] == "[REDACTED]"
        assert details["code"] == "[REDACTED]"
        assert details["prompt"] == "[REDACTED]"
        assert details["token"] == "[REDACTED]"
        # Non-sensitive values pass through.
        assert details["ok"] == "safe-value"

    def test_no_credential_like_values_leak(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger._info(
            event_type="policy_decision",
            capability="file_system",
            action="read_file",
            outcome="DENY",
            correlation_id="c1",
            details={
                "api_key": "sk-live-9f2b",
                "authorization": "Bearer token123",
                "reason": "safe reason",
            },
        )
        entry = _read_entries(audit_dir["log_path"])[-1]
        serialized = json.dumps(entry)
        assert "sk-live-9f2b" not in serialized
        assert "token123" not in serialized
        assert "safe reason" in serialized

    def test_bytes_and_arbitrary_objects_sanitized(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger._info(
            event_type="file_operation",
            capability="file_system",
            action="read",
            outcome="SUCCESS",
            correlation_id="c1",
            details={"raw": b"\x89PNG\r\n", "obj": object()},
        )
        entry = _read_entries(audit_dir["log_path"])[-1]
        details = entry["details"]
        assert details["raw"] == "[BYTES]"
        assert isinstance(details["obj"], str)

    def test_long_values_truncated(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger._info(
            event_type="tool_execution_end",
            capability="x",
            action="end",
            outcome="SUCCESS",
            correlation_id="c1",
            details={"note": "a" * 2000},
        )
        entry = _read_entries(audit_dir["log_path"])[-1]
        assert len(entry["details"]["note"]) <= 512 + len("...[TRUNCATED]")


# ---------------------------------------------------------------------------
# Correlation IDs
# ---------------------------------------------------------------------------


class TestCorrelationIds:
    def test_correlation_id_carried_through_events(self, audit_dir) -> None:
        logger = audit_dir["logger"]
        logger.log_policy_decision(
            capability="file_system",
            action="read_file",
            decision="ALLOW",
            reason="ok",
            correlation_id="task-9-call-0",
        )
        logger.log_tool_execution_start(
            tool_name="read_file",
            correlation_id="task-9-call-0",
        )
        logger.log_tool_execution_end(
            tool_name="read_file",
            correlation_id="task-9-call-0",
            success=True,
        )
        entries = _read_entries(audit_dir["log_path"])
        relevant = [
            e for e in entries if e.get("correlation_id") == "task-9-call-0"
        ]
        # policy_decision + tool_execution_start + tool_execution_end
        assert len(relevant) == 3
        for e in relevant:
            assert e["correlation_id"] == "task-9-call-0"


# ---------------------------------------------------------------------------
# Malformed events
# ---------------------------------------------------------------------------


class TestMalformedEvents:
    def test_non_json_serializable_details_does_not_crash(self, audit_dir) -> None:
        logger = audit_dir["logger"]

        class _Unserializable:
            def __str__(self) -> str:  # pragma: no cover - never called for cycles
                raise RuntimeError("boom")

        # A cyclic dict cannot be JSON-serialized. Logging must not raise.
        cyclic: dict[str, Any] = {}
        cyclic["self"] = cyclic
        try:
            logger._info(
                event_type="policy_decision",
                capability="file_system",
                action="read_file",
                outcome="DENY",
                correlation_id="c1",
                details=cyclic,
            )
        except Exception as exc:  # pragma: no cover - must not raise
            pytest.fail(f"Audit logging raised on malformed event: {exc}")

        # Logger must continue working after a malformed event.
        logger.log_policy_decision(
            capability="file_system",
            action="write_file",
            decision="DENY",
            reason="after malformed",
            correlation_id="c2",
        )
        entries = _read_entries(audit_dir["log_path"])
        assert any("after malformed" in json.dumps(e) for e in entries)


# ---------------------------------------------------------------------------
# Logger failure isolation
# ---------------------------------------------------------------------------


class TestLoggerFailureIsolation:
    def test_audit_failure_does_not_break_primary_operation(self, audit_dir) -> None:
        """A failing audit log write must not raise to the caller."""
        logger = audit_dir["logger"]

        # Break the underlying handler so every write fails.
        for handler in list(logger._logger.handlers):
            logger._logger.removeHandler(handler)
        logger._logger.addHandler(_FailingHandler())

        # None of these may raise.
        logger.log_policy_decision(
            capability="file_system", action="read_file",
            decision="ALLOW", reason="ok", correlation_id="c1",
        )
        logger.log_tool_execution_start(tool_name="read_file", correlation_id="c1")
        logger.log_tool_execution_end(
            tool_name="read_file", correlation_id="c1", success=True,
        )
        logger.log_sandbox_execution(correlation_id="c1", success=True)
        logger.log_file_operation(operation="read", correlation_id="c1", success=True)


class _FailingHandler(logging.Handler):
    """A logging handler whose emit always raises."""

    def emit(self, record: logging.LogRecord) -> None:
        raise OSError("simulated disk failure")


# ---------------------------------------------------------------------------
# Agent integration: policy DENY, success, failure logging
# ---------------------------------------------------------------------------


class _RecordingAuditLogger:
    """A test double that records audit calls instead of writing them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def log_policy_decision(self, **kwargs) -> None:
        self.calls.append(("policy_decision", kwargs))

    def log_tool_execution_start(self, **kwargs) -> None:
        self.calls.append(("tool_execution_start", kwargs))

    def log_tool_execution_end(self, **kwargs) -> None:
        self.calls.append(("tool_execution_end", kwargs))

    def log_sandbox_execution(self, **kwargs) -> None:
        self.calls.append(("sandbox_execution", kwargs))

    def log_file_operation(self, **kwargs) -> None:
        self.calls.append(("file_operation", kwargs))


@pytest.fixture
def recording_logger(monkeypatch):
    """Inject a recording audit logger into the agent module."""
    rec = _RecordingAuditLogger()
    monkeypatch.setattr("core.agent.agent.get_audit_logger", lambda: rec)
    return rec


def _make_agent(plan):
    from core.agent import Agent, AgentConfig, Plan, ToolDefinition
    from core.agent.registry import DefaultToolRegistry
    from core.routing import Capability
    from core.security.policy_engine import init_policy_engine

    tool_registry = DefaultToolRegistry()
    tool_registry.register(
        ToolDefinition(
            name="read_file", description="read", capability="file_system"
        )
    )
    executor = _FakeExecutor(tool_registry)
    agent = Agent(
        model_router=_FakeRouter(),
        model_gateway=_FakeModelGateway(),
        tool_executor=executor,
        planner=_FakePlanner(plan=plan),
        verifier=_FakeVerifier(),
        config=AgentConfig(max_iterations=5),
        tool_registry=tool_registry,
    )
    return agent, executor


class _FakeExecutor:
    def __init__(self, registry) -> None:
        self._registry = registry
        self.calls = []

    @property
    def registry(self):
        return self._registry

    async def execute(self, call):
        from core.agent.errors import UnknownToolError
        from core.agent.types import ToolResult
        self.calls.append(call)
        if not self._registry.has(call.tool_name):
            raise UnknownToolError(call.tool_name)
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output={"ok": True},
            error=False,
            error_message="",
            latency_ms=1.0,
        )


class _FakePlanner:
    def __init__(self, plan) -> None:
        self._plan = plan

    async def plan(self, task, available_tools, state):
        return self._plan


class _FakeVerifier:
    async def verify(self, state):
        from core.agent.types import VerificationResult
        return VerificationResult(passed=True, reason="OK")


class _FakeRouter:
    def route(self, request):
        from core.routing import Capability
        return type(
            "RoutingDecision",
            (),
            {
                "model": type(
                    "ModelDef",
                    (),
                    {
                        "logical_name": "general",
                        "provider": "ollama",
                        "provider_model": "qwen3:4b",
                        "capabilities": frozenset({Capability.GENERAL}),
                        "priority": 10,
                    },
                )(),
                "reason": "fake",
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

    async def generate(self, request):
        from core.llm.types import GenerationResponse

        return GenerationResponse(
            content="fake response",
            model=request.model or "fake-model",
        )


class TestAgentAuditIntegration:
    def _plan_single(self, tool_name: str, **inputs):
        from core.agent import Plan, PlanStep
        return Plan(
            goal="g",
            steps=(PlanStep(
                step_id="s0", description="do", tool_name=tool_name, inputs=inputs
            ),),
        )

    @pytest.mark.asyncio
    async def test_denied_tool_logs_policy_decision(self, recording_logger, monkeypatch):
        from core.security.policy_engine import init_policy_engine, reset_policy_engine
        try:
            init_policy_engine(config={
                "file_system": {"read_file": "DENY"},
                "default": "DENY",
            })
            agent, executor = _make_agent(
                self._plan_single("read_file", path="/etc/shadow")
            )
            result = await agent.run("read secret")
            assert result.status == "complete"
            assert len(executor.calls) == 0
            policy_calls = [
                c for (name, c) in recording_logger.calls
                if name == "policy_decision"
            ]
            assert len(policy_calls) == 1
            assert policy_calls[0]["decision"] == "DENY"
            assert policy_calls[0]["capability"] == "file_system"
            assert policy_calls[0]["action"] == "read_file"
        finally:
            reset_policy_engine()

    @pytest.mark.asyncio
    async def test_successful_execution_logs_start_and_end(
        self, recording_logger, monkeypatch
    ):
        from core.security.policy_engine import init_policy_engine, reset_policy_engine
        try:
            init_policy_engine()  # read_file is ALLOW by default
            agent, executor = _make_agent(
                self._plan_single("read_file", path="notes.txt")
            )
            result = await agent.run("read notes")
            assert result.status == "complete"
            assert len(executor.calls) == 1
            names = [n for (n, _) in recording_logger.calls]
            assert "tool_execution_start" in names
            assert "tool_execution_end" in names
            end_calls = [
                c for (n, c) in recording_logger.calls if n == "tool_execution_end"
            ]
            assert end_calls[-1]["success"] is True
        finally:
            reset_policy_engine()

    @pytest.mark.asyncio
    async def test_failed_execution_logs_failure(self, recording_logger, monkeypatch):
        from core.agent.types import ToolResult
        from core.security.policy_engine import init_policy_engine, reset_policy_engine

        class _FailingExecutor(_FakeExecutor):
            async def execute(self, call):
                self.calls.append(call)
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    output=None,
                    error=True,
                    error_message="Something failed",
                    latency_ms=1.0,
                )

        try:
            init_policy_engine()  # read_file ALLOW by default
            agent, executor = _make_agent(self._plan_single("read_file", path="x.txt"))
            # Replace executor with a failing one.
            agent._executor = _FailingExecutor(executor._registry)
            result = await agent.run("read that fails")
            # Unrecovered tool failure surfaces as FAILED (integration-hardening
            # correction); the audit logger still records the failure.
            assert result.status == "failed"
            end_calls = [
                c for (n, c) in recording_logger.calls if n == "tool_execution_end"
            ]
            assert end_calls
            assert end_calls[-1]["success"] is False
        finally:
            reset_policy_engine()

    @pytest.mark.asyncio
    async def test_correlation_id_links_policy_and_execution(
        self, recording_logger, monkeypatch
    ):
        from core.security.policy_engine import init_policy_engine, reset_policy_engine
        try:
            init_policy_engine()
            agent, executor = _make_agent(
                self._plan_single("read_file", path="notes.txt")
            )
            await agent.run("read notes")
            # Find the tool_execution_start correlation id and confirm the
            # matching policy_decision uses the same id.
            start = [
                c for (n, c) in recording_logger.calls if n == "tool_execution_start"
            ][0]
            cid = start["correlation_id"]
            policy = [
                c for (n, c) in recording_logger.calls if n == "policy_decision"
            ][0]
            assert policy["correlation_id"] == cid
        finally:
            reset_policy_engine()
