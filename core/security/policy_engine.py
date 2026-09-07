"""Centralized security Policy Engine for SovereignAI Workbench.

This module provides a deterministic, framework-independent policy engine that
evaluates agent/tool actions **before execution**.  It supports ``ALLOW``,
``DENY``, and ``REQUIRE_APPROVAL`` decisions based on capability/action pairs.

Design notes
------------

* Policies are **capability/action based** — never model/provider based.
* The engine is deterministic: no randomness, no cloud/API dependency.
* Decisions are returned as :class:`PolicyDecision` objects; the agent turns a
  non-``ALLOW`` result into a structured, safe denial observation.
* Workspace/path boundaries are **not** duplicated here — they remain the
  responsibility of :class:`core.tools.workspace.Workspace` and the tools that
  delegate to it.  The policy engine only decides *whether* a tool may run.
* Configuration is loaded from project configuration (``config/policy.yaml``)
  with secure defaults that deny unknown/dangerous actions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

from core.agent.types import ToolCall

logger = logging.getLogger("sovereign-ai.security.policy")


#: Valid decision strings.  Kept as plain strings so config stays JSON/YAML-safe.
ALLOW = "ALLOW"
DENY = "DENY"
REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class PolicyDecision:
    """The structured result of a single policy evaluation.

    A ``PolicyDecision`` says what the capability/action should do and why.
    It is deliberately free of any sensitive material: no paths, prompts,
    credentials, or internal exception text.
    """

    __slots__ = ("decision", "reason", "data", "capability", "action")

    def __init__(
        self,
        decision: Literal["ALLOW", "DENY", "REQUIRE_APPROVAL"],
        reason: str,
        data: Optional[dict[str, Any]] = None,
        capability: Optional[str] = None,
        action: Optional[str] = None,
    ) -> None:
        self.decision = decision
        self.reason = reason
        self.data = data or {}
        self.capability = capability
        self.action = action

    def is_allowed(self) -> bool:
        return self.decision == ALLOW

    def is_denied(self) -> bool:
        return self.decision == DENY

    def requires_approval(self) -> bool:
        return self.decision == REQUIRE_APPROVAL

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"PolicyDecision(decision={self.decision!r}, reason={self.reason!r})"


class PolicyEngine:
    """Centralized, deterministic policy engine for evaluating tool actions.

    The engine is constructed with a configuration mapping:
    ``capability -> {action -> decision}`` plus a ``"default"`` entry.
    A user config is **merged over** the secure defaults so partial
    configuration is safe (nothing is silently dropped).

    The engine is immutable once constructed, except through the explicit
    :meth:`update_policy` mutation point.  It is safe to share a single
    instance across concurrent agent runs.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        # Merge user config over the secure defaults so partial configs
        # preserve the default decisions for unspecified actions.
        self.config = dict(self._default_config())
        if config is not None:
            self._merge_config(self.config, config)
        self._validate(self.config)
        logger.info(
            "policy.initialized  capabilities=%s",
            sorted(k for k in self.config if k != "default"),
        )

    # ------------------------------------------------------------------ config

    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Secure default configuration — deny dangerous/unknown actions.

        The default allows only the *existing legitimate workflows* of the
        workbench (files, documents, RAG search, vision/OCR) and denies code
        execution plus any unknown capability/action.
        """
        return {
            # File system tools (Phase 5A).
            "file_system": {
                "list_files": ALLOW,
                "read_file": ALLOW,
                "write_file": ALLOW,
            },
            # Document generation tool (Phase 5E).
            "document_generation": {
                "create_document": ALLOW,
            },
            # Local RAG / knowledge base tool (Phase 5B).
            "knowledge": {
                "search_knowledge_base": ALLOW,
            },
            # Vision / OCR tools (Phase 5C).
            "vision": {
                "ocr_image": ALLOW,
                "analyze_image": ALLOW,
            },
            # Sandbox code execution (Phase 5D) — DANGEROUS, denied by default.
            "code_execution": {
                "execute_code": DENY,
            },
            # Anything not explicitly listed above is denied.
            "default": DENY,
        }

    @staticmethod
    def _merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> None:
        """Recursively merge ``overlay`` into ``base`` in place.

        Nested mappings are merged key-by-key so a partial user config keeps
        the default decisions for actions it does not mention.  Top-level
        scalars (e.g. the ``default`` decision) simply overwrite.
        """
        for key, value in overlay.items():
            if (
                isinstance(value, dict)
                and isinstance(base.get(key), dict)
            ):
                PolicyEngine._merge_config(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _validate(config: dict[str, Any]) -> None:
        """Ensure every decision is one of the allowed values."""
        for cap, actions in config.items():
            if cap == "default":
                continue
            if not isinstance(actions, dict):
                raise ValueError(
                    f"Policy config entry {cap!r} must be a mapping"
                )
            for action, decision in actions.items():
                if decision not in (ALLOW, DENY, REQUIRE_APPROVAL):
                    raise ValueError(
                        f"Invalid decision {decision!r} for "
                        f"{cap}.{action}; expected ALLOW, DENY, or "
                        "REQUIRE_APPROVAL"
                    )

    # ------------------------------------------------------------------ eval

    def evaluate(self, tool_call: ToolCall) -> PolicyDecision:
        """Evaluate a tool call and return a structured decision.

        Evaluations are deterministic and never raise on malformed input —
        an unrecognised capability/action falls through to the ``default``
        (which is ``DENY`` unless configured otherwise).
        """
        capability, action = self._extract_capability_action(tool_call)
        decision = self._lookup(capability, action)

        logger.info(
            "policy.eval  tool=%s  capability=%s  action=%s  decision=%s",
            tool_call.tool_name,
            capability,
            action,
            decision,
        )

        if decision == ALLOW:
            return PolicyDecision(
                ALLOW,
                f"Allowed by policy: {capability}.{action}",
                capability=capability,
                action=action,
            )
        if decision == REQUIRE_APPROVAL:
            return PolicyDecision(
                REQUIRE_APPROVAL,
                f"Approval required: {capability}.{action}",
                {"requires_approval": True, "capability": capability, "action": action},
                capability=capability,
                action=action,
            )
        return PolicyDecision(
            DENY,
            f"Denied by policy: {capability}.{action}",
            {"capability": capability, "action": action},
            capability=capability,
            action=action,
        )

    def _lookup(self, capability: str, action: str) -> str:
        """Return the effective decision for a capability/action."""
        cap_rules = self.config.get(capability)
        if isinstance(cap_rules, dict) and action in cap_rules:
            return cap_rules[action]
        return self.config.get("default", DENY)

    # ------------------------------------------------------------------ config access

    def get_policy(self, capability: str, action: str) -> str:
        """Return the current decision for a capability/action (no fallback merge)."""
        return self._lookup(capability, action)

    def update_policy(self, capability: str, action: str, decision: str) -> None:
        """Mutate a single rule at runtime.

        This is the only supported mutation point.  The engine is otherwise
        immutable, keeping evaluations deterministic.
        """
        if decision not in (ALLOW, DENY, REQUIRE_APPROVAL):
            raise ValueError(
                f"Invalid decision {decision!r}; expected ALLOW, DENY, or "
                "REQUIRE_APPROVAL"
            )
        if capability not in self.config:
            self.config[capability] = {}
        self.config[capability][action] = decision
        logger.info(
            "policy.updated  capability=%s  action=%s  decision=%s",
            capability,
            action,
            decision,
        )

    def rules_count(self) -> int:
        """Return the number of explicit capability/action rules."""
        return sum(
            len(actions)
            for capability, actions in self.config.items()
            if capability != "default" and isinstance(actions, dict)
        )

    # ------------------------------------------------------------------ tool mapping

    #: maps registered tool names → (capability, action).  This is the single
    #: source of truth for the tools the workbench exposes to the agent.
    _TOOL_CAPABILITY = {
        # file tools (Phase 5A)
        "list_files": ("file_system", "list_files"),
        "read_file": ("file_system", "read_file"),
        "write_file": ("file_system", "write_file"),
        # document tool (Phase 5E)
        "create_document": ("document_generation", "create_document"),
        # RAG tool (Phase 5B)
        "search_knowledge_base": ("knowledge", "search_knowledge_base"),
        # sandbox code execution (Phase 5D)
        "execute_code": ("code_execution", "execute_code"),
        # vision/OCR tools (Phase 5C)
        "ocr_image": ("vision", "ocr_image"),
        "analyze_image": ("vision", "analyze_image"),
    }

    def _extract_capability_action(self, tool_call: ToolCall) -> tuple[str, str]:
        """Map a tool name to a (capability, action) pair.

        Recognised tools use their canonical capability/action.  Unknown tool
        names map to a best-effort ``(capability, tool_name)`` based on name
        heuristics, so unknown-but-descriptive tools still land on a sensible
        policy bucket and default to DENY unless explicitly allowed.
        """
        tool_name = tool_call.tool_name or ""
        mapped = self._TOOL_CAPABILITY.get(tool_name)
        if mapped is not None:
            return mapped

        lowered = tool_name.lower()
        if any(k in lowered for k in ("file", "read", "write", "list")):
            return ("file_system", tool_name)
        if "document" in lowered:
            return ("document_generation", tool_name)
        if any(k in lowered for k in ("knowledge", "search", "query", "rag")):
            return ("knowledge", tool_name)
        if any(k in lowered for k in ("execute", "code", "shell", "sandbox", "run")):
            return ("code_execution", tool_name)
        if any(k in lowered for k in ("image", "ocr", "vision", "analyze", "photo")):
            return ("vision", tool_name)
        return ("unknown", tool_name)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_policy_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load a policy config from YAML, returning the merged effective config.

    The on-disk YAML is user configuration with this shape::

        default: DENY
        file_system:
          read_file: ALLOW
          write_file: REQUIRE_APPROVAL
        code_execution:
          execute_code: DENY

    The result is merged over the secure defaults, so keys not present in the
    file keep their default decisions.

    Args:
        path: Path to the YAML config file.  Defaults to ``config/policy.yaml``
            relative to the project root.

    Returns:
        The merged, validated configuration mapping.
    """
    if path is None:
        # Resolve relative to the project root (the directory containing
        # ``pytest.ini`` / the ``config`` folder), independent of CWD.
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "config" / "policy.yaml"

    if not path.is_file():
        logger.debug("No policy config at %s; using secure defaults", path)
        return dict(PolicyEngine._default_config())

    import yaml  # local import keeps the engine dependency-free at call time

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    engine = PolicyEngine(config=dict(raw))
    return dict(engine.config)


# ---------------------------------------------------------------------------
# Global instance (application wiring)
# ---------------------------------------------------------------------------
#
# The engine is a singleton so every agent in the process shares one policy.
# ``init_policy_engine`` is the only way to install it; ``reset_policy_engine``
# is provided for tests.  When no engine is installed, the agent runs with
# policy enforcement disabled (backward compatible).

_policy_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> Optional[PolicyEngine]:
    """Return the process-wide policy engine, if installed.

    Returns ``None`` when no policy engine has been initialised — the agent
    then runs without policy enforcement (backward-compatible behaviour).
    """
    return _policy_engine


def init_policy_engine(
    config: Optional[dict[str, Any]] = None,
    *,
    config_path: Optional[Path] = None,
) -> PolicyEngine:
    """Install and return the process-wide policy engine.

    If ``config`` is given it is merged over the secure defaults.  Otherwise
    the engine loads ``config/policy.yaml`` (if present) over the defaults.

    Args:
        config: Optional in-memory policy configuration (merged over defaults).
        config_path: Optional explicit path to a YAML policy config.

    Returns:
        The installed :class:`PolicyEngine`.
    """
    global _policy_engine
    if config is not None:
        _policy_engine = PolicyEngine(config=config)
    else:
        _policy_engine = PolicyEngine(config=load_policy_config(config_path))
    return _policy_engine


def reset_policy_engine() -> None:
    """Remove the process-wide policy engine (used by tests).

    After a reset the agent runs without policy enforcement, preserving the
    pre-Phase-6 behaviour for callers that do not opt in.
    """
    global _policy_engine
    _policy_engine = None