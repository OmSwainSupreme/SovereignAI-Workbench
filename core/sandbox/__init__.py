"""SovereignAI Workbench — Secure Code Sandbox (Phase 5D).

Framework-agnostic, isolated code-execution environment.

Architecture
------------

The sandbox package is layered:

* :mod:`core.sandbox.types`     — plain dataclasses for ExecutionRequest
                                  and ExecutionResult. No framework
                                  dependencies.
* :mod:`core.sandbox.limits`    — :class:`ResourceLimits` and
                                  :class:`SandboxConfig` dataclasses that
                                  control isolation, timeouts, and capacity.
* :mod:`core.sandbox.errors`    — exception hierarchy for sandbox errors.
* :mod:`core.sandbox.interface` — :class:`Sandbox` ABC. The only
                                  abstraction the agent / tool layer needs.
* :mod:`core.sandbox.docker_sandbox` — :class:`DockerSandbox`,
                                  the first concrete implementation,
                                  providing OS-/container-level isolation.
* :mod:`core.sandbox.tools`     — :func:`register_code_tools` which
                                  registers an ``execute_code``
                                  :class:`ToolDefinition` into a
                                  :class:`DefaultToolRegistry`.

Security model
--------------

Arbitrary AI-generated code is treated as **untrusted**. The primary
security boundary is process/container/OS-level isolation provided by
the :class:`DockerSandbox` backend. Python-level restrictions are NEVER
the primary defence.

Key properties of the default configuration:

* No host filesystem mounts — the container has only a temporary
  in-container workspace.
* No Docker socket mount.
* No ``--privileged`` mode, no host network/PID/IPC namespace.
* Networking is disabled at the container level.
* The container runs as a non-root user.
* The container is removed after each execution.
* The container receives a minimal, allowlisted environment.
* The host user's environment is NOT inherited.
* All stdout/stderr is bounded.
* Hard execution timeout is enforced by the host and the container.

Logging policy
--------------

* Source code is never logged.
* stdout/stderr contents are never logged.
* Environment values are never logged.
* Only safe metadata (execution id, language, duration, exit code,
  timeout status, success/failure, output truncation status) is logged.

The sandbox does not depend on FastAPI, Pydantic, the Agent runtime,
the Model Router, the Ollama provider, or any external API. It is
independently testable.
"""
from __future__ import annotations

from core.sandbox.errors import (
    SandboxError,
    SandboxBackendError,
    SandboxConfigError,
    SandboxValidationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    UnsupportedLanguageError,
)
from core.sandbox.limits import (
    ResourceLimits,
    SandboxConfig,
)
from core.sandbox.types import (
    ExecutionRequest,
    ExecutionResult,
    LANGUAGE_PYTHON,
)
from core.sandbox.interface import Sandbox, _NullSandbox
from core.sandbox.tools import (
    EXECUTE_CODE_TOOL,
    CodeToolExecutor,
    register_code_tools,
)

__all__ = [
    # errors
    "SandboxError",
    "SandboxBackendError",
    "SandboxConfigError",
    "SandboxValidationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    "UnsupportedLanguageError",
    # config
    "ResourceLimits",
    "SandboxConfig",
    # types
    "ExecutionRequest",
    "ExecutionResult",
    "LANGUAGE_PYTHON",
    # interface
    "Sandbox",
    "_NullSandbox",
    # tools
    "EXECUTE_CODE_TOOL",
    "CodeToolExecutor",
    "register_code_tools",
]
