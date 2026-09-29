"""Unit tests for the centralized security Policy Engine (Phase 6)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.agent.types import ToolCall
from core.security.policy_engine import (
    ALLOW,
    DENY,
    REQUIRE_APPROVAL,
    PolicyDecision,
    PolicyEngine,
    get_policy_engine,
    init_policy_engine,
    reset_policy_engine,
)


def _call(tool_name: str, **arguments) -> ToolCall:
    return ToolCall(call_id=f"test-{tool_name}", tool_name=tool_name, arguments=arguments)


class TestDefaultConfig:
    """Secure defaults must deny dangerous/unknown actions."""

    def test_default_denies_unknown(self) -> None:
        engine = PolicyEngine()
        assert engine.config["default"] == DENY

    def test_default_allows_existing_legitimate_tools(self) -> None:
        engine = PolicyEngine()
        # Existing legitimate workflows continue under the default policy.
        for tool in (
            "list_files", "read_file", "write_file",
            "create_document", "search_knowledge_base",
            "ocr_image", "analyze_image",
        ):
            assert engine.evaluate(_call(tool)).is_allowed()

    def test_default_denies_code_execution(self) -> None:
        engine = PolicyEngine()
        assert engine.evaluate(_call("execute_code", code="print(1)")).is_denied()

    def test_unknown_tool_denied_by_default(self) -> None:
        engine = PolicyEngine()
        decision = engine.evaluate(_call("totally_unknown_tool_xyz"))
        assert decision.is_denied()
        assert decision.decision == DENY


class TestPolicyDecision:
    """ALLOW / DENY / REQUIRE_APPROVAL semantics."""

    def test_decision_methods(self) -> None:
        assert PolicyDecision(ALLOW, "ok").is_allowed()
        assert not PolicyDecision(ALLOW, "ok").is_denied()
        assert PolicyDecision(DENY, "no").is_denied()
        assert not PolicyDecision(DENY, "no").is_allowed()
        assert PolicyDecision(REQUIRE_APPROVAL, "ask").requires_approval()
        assert not PolicyDecision(REQUIRE_APPROVAL, "ask").is_allowed()

    def test_reason_is_safe_and_structured(self) -> None:
        d = PolicyEngine().evaluate(_call("execute_code", code="secret"))
        assert d.decision == DENY
        assert "secret" not in d.reason
        assert "/tmp" not in d.reason
        assert "localhost" not in d.reason
        assert d.data["capability"] == "code_execution"
        assert d.data["action"] == "execute_code"


class TestEvaluate:
    """Capability/action based evaluation."""

    def test_mapping_covers_registered_tools(self) -> None:
        engine = PolicyEngine()
        cases = {
            "list_files": ("file_system", "list_files"),
            "read_file": ("file_system", "read_file"),
            "write_file": ("file_system", "write_file"),
            "create_document": ("document_generation", "create_document"),
            "search_knowledge_base": ("knowledge", "search_knowledge_base"),
            "execute_code": ("code_execution", "execute_code"),
            "ocr_image": ("vision", "ocr_image"),
            "analyze_image": ("vision", "analyze_image"),
        }
        for tool, (cap, action) in cases.items():
            got_cap, got_action = engine._extract_capability_action(_call(tool))
            assert (got_cap, got_action) == (cap, action), tool

    def test_unknown_tool_name_heuristic(self) -> None:
        engine = PolicyEngine()
        # Descriptive but unregistered names land on a policy bucket.
        assert engine._extract_capability_action(_call("read_notes"))[0] == "file_system"
        assert engine._extract_capability_action(_call("vision_histogram"))[0] == "vision"
        assert engine._extract_capability_action(_call("rag_retrieve"))[0] == "knowledge"
        assert engine._extract_capability_action(_call("run_shell"))[0] == "code_execution"

    def test_malformed_tool_call_returns_deny(self) -> None:
        engine = PolicyEngine()
        # None/empty tool name must not crash — falls to default (deny).
        decision = engine.evaluate(_call(""))
        assert decision.is_denied()


class TestCustomConfig:
    """Partial configs merge over defaults; decisions can be flipped."""

    def test_custom_deny_overrides_default_allow(self) -> None:
        engine = PolicyEngine(config={
            "file_system": {"read_file": DENY},
        })
        assert engine.evaluate(_call("read_file")).is_denied()
        # Untouched default actions still allowed.
        assert engine.evaluate(_call("write_file")).is_allowed()

    def test_custom_allow_overrides_default_deny(self) -> None:
        engine = PolicyEngine(config={
            "code_execution": {"execute_code": ALLOW},
        })
        assert engine.evaluate(_call("execute_code")).is_allowed()

    def test_require_approval_decision(self) -> None:
        engine = PolicyEngine(config={
            "file_system": {"write_file": REQUIRE_APPROVAL},
        })
        decision = engine.evaluate(_call("write_file"))
        assert decision.requires_approval()
        assert decision.data["requires_approval"] is True

    def test_invalid_decision_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid decision"):
            PolicyEngine(config={"file_system": {"read_file": "MAYBE"}})


class TestUpdatePolicy:
    """Runtime mutation point."""

    def test_update_flips_decision(self) -> None:
        engine = PolicyEngine()
        assert engine.get_policy("file_system", "read_file") == ALLOW
        engine.update_policy("file_system", "read_file", DENY)
        assert engine.get_policy("file_system", "read_file") == DENY
        assert engine.evaluate(_call("read_file")).is_denied()

    def test_update_deny_to_approval(self) -> None:
        engine = PolicyEngine()
        engine.update_policy("file_system", "write_file", REQUIRE_APPROVAL)
        assert engine.evaluate(_call("write_file")).requires_approval()

    def test_update_invalid_decision(self) -> None:
        engine = PolicyEngine()
        with pytest.raises(ValueError, match="Invalid decision"):
            engine.update_policy("file_system", "read_file", "NOPE")

    def test_update_new_capability(self) -> None:
        engine = PolicyEngine()
        engine.update_policy("custom_cap", "run_task", ALLOW)
        assert engine.get_policy("custom_cap", "run_task") == ALLOW
        # Registers a new rule alongside defaults; existing rules untouched.
        assert engine.get_policy("file_system", "read_file") == ALLOW


class TestPolicyEngineSingletons:
    """Global install/reset lifecycle."""

    def test_globals_roundtrip(self) -> None:
        reset_policy_engine()
        assert get_policy_engine() is None
        engine = init_policy_engine()
        assert get_policy_engine() is engine
        reset_policy_engine()
        assert get_policy_engine() is None

    def test_init_with_custom_config(self) -> None:
        reset_policy_engine()
        engine = init_policy_engine(config={"code_execution": {"execute_code": ALLOW}})
        assert engine.evaluate(_call("execute_code")).is_allowed()
        reset_policy_engine()

    def test_default_config_path_serves_default_when_no_file(self) -> None:
        # Simulate no config/policy.yaml present: init_policy_engine() falls
        # back to secure defaults.
        reset_policy_engine()
        engine = init_policy_engine()
        assert engine.evaluate(_call("unknown_thing")).is_denied()
        reset_policy_engine()


class TestYAMLConfig:
    """Configuration loaded from project YAML."""

    def test_load_policy_config_from_path(self) -> None:
        from core.security.policy_engine import load_policy_config
        td = tempfile.mkdtemp()
        p = Path(td) / "policy.yaml"
        p.write_text(
            "default: DENY\n"
            "code_execution:\n"
            "  execute_code: ALLOW\n"
            "file_system:\n"
            "  write_file: REQUIRE_APPROVAL\n",
            encoding="utf-8",
        )
        cfg = load_policy_config(p)
        eng = PolicyEngine(config=cfg)
        assert eng.evaluate(_call("execute_code")).is_allowed()
        assert eng.evaluate(_call("write_file")).requires_approval()
        # Untouched defaults remain.
        assert eng.evaluate(_call("read_file")).is_allowed()

    def test_missing_file_returns_defaults(self) -> None:
        from core.security.policy_engine import load_policy_config
        td = tempfile.mkdtemp()
        cfg = load_policy_config(Path(td) / "does_not_exist.yaml")
        eng = PolicyEngine(config=cfg)
        assert eng.evaluate(_call("read_file")).is_allowed()
        assert eng.evaluate(_call("execute_code")).is_denied()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])