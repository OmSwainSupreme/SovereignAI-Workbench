"""Framework-agnostic data types for the Secure Code Sandbox.

These types are deliberately plain dataclasses — they must not depend on
FastAPI, Pydantic, or any other application framework. The application layer
may wrap them in framework-specific DTOs for transport.

These types are the ONLY objects that cross the sandbox boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------


LANGUAGE_PYTHON = "python"
"""The only language supported in this phase."""


# ---------------------------------------------------------------------------
# Execution request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionRequest:
    """A request to execute code inside the sandbox.

    Attributes:
        execution_id: Unique identifier for this execution. Used in logs and
            for correlation with audit records. Must be unique per call.
        language: The programming language of ``code``. Currently only
            ``"python"`` is supported.
        code: The source code to execute. Must be valid UTF-8 text.
        timeout_seconds: Maximum wall-clock time the sandbox will wait before
            forcibly terminating the execution. Must not exceed the configured
            maximum.
        stdin: Optional string to pass as stdin to the program.
        resource_limits: Optional overrides for resource limits. Values that
            are None or exceed the configured maximum are ignored; the
            configured maximum is used instead.
    """

    execution_id: str
    language: str
    code: str
    timeout_seconds: float = 30.0
    stdin: Optional[str] = None
    resource_limits: Optional[dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.execution_id:
            raise ValueError("execution_id must be a non-empty string")
        if not self.language:
            raise ValueError("language must be a non-empty string")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def _unsafe(
        cls,
        execution_id: str,
        language: str,
        code: str,
        timeout_seconds: float,
        stdin: Optional[str] = None,
        resource_limits: Optional[dict] = None,
    ) -> "ExecutionRequest":
        """Construct a request without post-init validation.

        Internal helper for tests and adapters that need to construct
        a request that would fail basic validation (e.g. negative
        timeout) so the sandbox's own validation can be exercised.
        Normal callers should never use this.
        """
        obj = cls.__new__(cls)
        object.__setattr__(obj, "execution_id", execution_id)
        object.__setattr__(obj, "language", language)
        object.__setattr__(obj, "code", code)
        object.__setattr__(obj, "timeout_seconds", timeout_seconds)
        object.__setattr__(obj, "stdin", stdin)
        object.__setattr__(obj, "resource_limits", resource_limits or {})
        return obj


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionResult:
    """The result of a sandboxed code execution.

    This type carries structured information about the execution outcome.
    ``stdout`` and ``stderr`` are captured output (possibly truncated). No
    absolute host paths or sensitive data are included in the result.

    The ``output_truncated`` field indicates whether stdout or stderr was
    truncated due to the configured output size limit. Callers that parse
    structured output (e.g. JSON) should be aware that truncation may
    break the structure.
    """

    success: bool
    """True if the program exited with code 0 and no timeout occurred."""

    exit_code: int
    """The program exit code. 0 typically means success (language-dependent)."""

    stdout: str
    """Captured stdout. Empty string if no output was produced. May be truncated."""

    stderr: str
    """Captured stderr. Empty string if no error output was produced. May be truncated."""

    timed_out: bool
    """True if the execution exceeded the configured timeout."""

    duration_seconds: float
    """Wall-clock time from execution start to result availability, in seconds."""

    output_truncated: bool = False
    """True if stdout or stderr was truncated to respect output limits."""

    execution_id: str = ""
    """Echo of the request's execution_id for correlation."""

    language: str = ""
    """Echo of the request's language for audit purposes."""

    @property
    def truncated_stdout(self) -> bool:
        """True if stdout was truncated (convenience alias)."""
        return self.output_truncated

    @property
    def truncated_stderr(self) -> bool:
        """True if stderr was truncated (convenience alias)."""
        return self.output_truncated

    def to_dict(self) -> dict:
        """Convert to a plain dict for JSON serialization.

        Suitable for use in :class:`core.agent.types.ToolResult` output.
        """
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "output_truncated": self.output_truncated,
            "execution_id": self.execution_id,
            "language": self.language,
        }
