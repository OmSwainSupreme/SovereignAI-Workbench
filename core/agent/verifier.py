"""Built-in verifiers for the Agent Runtime.

A verifier inspects the agent's final state and returns a pass/fail verdict.
This module provides :class:`SimpleVerifier`, a minimal implementation that
returns "pass" by default and can be configured to fail under specific
conditions.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.agent.interfaces import Verifier
from core.agent.types import AgentState, VerificationResult


_logger = logging.getLogger("sovereign-ai.agent.verifier")


class SimpleVerifier(Verifier):
    """A minimal verifier that returns a pass/fail based on simple heuristics.

    The verifier passes the agent's output if any of these are true:

    * The agent produced at least one observation or one artifact.
    * The agent's iteration is at least 1 (i.e. it actually did work).

    Otherwise, the verifier fails with a "no progress" message.

    The verifier can be configured to always pass, always fail, or fail
    under specific conditions.
    """

    def __init__(
        self,
        always_pass: bool = False,
        always_fail: bool = False,
        require_artifacts: bool = False,
        require_observations: bool = False,
    ) -> None:
        """
        Args:
            always_pass: If True, always return ``passed=True``.
            always_fail: If True, always return ``passed=False``.
            require_artifacts: If True, fail if no artifacts were produced.
            require_observations: If True, fail if no observations were made.
        """
        if always_pass and always_fail:
            raise ValueError("always_pass and always_fail are mutually exclusive")
        self._always_pass = always_pass
        self._always_fail = always_fail
        self._require_artifacts = require_artifacts
        self._require_observations = require_observations

    async def verify(self, state: AgentState) -> VerificationResult:
        if self._always_pass:
            return VerificationResult(passed=True, reason="Verifier configured to always pass.")

        if self._always_fail:
            return VerificationResult(
                passed=False,
                reason="Verifier configured to always fail.",
                suggestions=("This is a test fixture; configure a different verifier.",),
            )

        if self._require_artifacts and not state.artifacts:
            return VerificationResult(
                passed=False,
                reason="No artifacts produced.",
                suggestions=("Add tools that create artifacts, or relax the requirement.",),
            )

        if self._require_observations and not state.observations:
            return VerificationResult(
                passed=False,
                reason="No observations made.",
                suggestions=("Add tools that return observations.",),
            )

        if not state.artifacts and not state.observations and state.iteration == 0:
            return VerificationResult(
                passed=False,
                reason="Agent made no progress.",
                suggestions=("The agent may not have run any iterations.",),
            )

        return VerificationResult(
            passed=True,
            reason="Agent produced at least one observation or artifact.",
        )
