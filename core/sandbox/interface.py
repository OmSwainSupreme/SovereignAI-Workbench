"""Abstract :class:`Sandbox` interface.

The :class:`Sandbox` is the only abstraction the agent / tool layer
needs to know about. Concrete backends (Docker, Linux namespaces, a
restricted local runner, etc.) implement this interface.

The interface is intentionally small: one method,
:meth:`Sandbox.execute`, that takes an :class:`ExecutionRequest` and
returns an :class:`ExecutionResult`. The interface does not expose
host paths, environment values, or any other implementation detail.

Backends MUST:

* Validate the request before execution. A :class:`SandboxValidationError`
  is raised for any invalid request (unsupported language, empty
  code, oversized code, invalid timeout, etc.).
* Enforce the configured timeout by actually terminating the
  process/container. They MUST NOT merely stop waiting for it.
* Capture stdout/stderr to bounded buffers, truncating safely if the
  configured output limit is exceeded.
* Not log source code, stdout contents, stderr contents, environment
  values, or absolute host paths.
* Be safe to construct without side effects. The :class:`DockerSandbox`
  may do a one-time sanity check at construction (verifying that
  Docker is installed) but must not start a container.
"""
from __future__ import annotations

import abc
import logging
from typing import Optional

from core.sandbox.limits import ResourceLimits, SandboxConfig
from core.sandbox.types import ExecutionRequest, ExecutionResult


_logger = logging.getLogger("sovereign-ai.sandbox.interface")


class Sandbox(abc.ABC):
    """An abstract isolated code-execution environment.

    Subclasses implement a specific isolation technology (Docker,
    Linux namespaces, microVMs, restricted local runner, etc.). The
    agent and tool layer use the abstract interface exclusively, so
    the backend can be swapped without changing agent code.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        """Initialise the sandbox with a :class:`SandboxConfig`.

        The default configuration is safe: no network, read-only root,
        non-root user, allowlisted environment.
        """
        self._config = config or SandboxConfig()

    # ------------------------------------------------------------------ Public

    @property
    def config(self) -> SandboxConfig:
        """The :class:`SandboxConfig` for this sandbox."""
        return self._config

    @property
    def backend_name(self) -> str:
        """A short identifier for the backend, e.g. ``"docker"``.

        Used for logging and audit records. Subclasses override this.
        """
        return self.__class__.__name__.lower()

    @abc.abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute ``request.code`` in the sandbox and return the result.

        The contract is:

        * The request is validated first; invalid requests raise
          :class:`SandboxValidationError` (or a subclass) and no
          container is started.
        * On success (exit code 0, no timeout), ``success=True`` is
          returned with the captured output.
        * On a non-zero exit code, ``success=False`` is returned; this
          is NOT an error — the program ran and produced a result.
        * On a timeout, ``timed_out=True`` is returned. The container
          is terminated before the function returns.
        * The execution is actually terminated: the container is
          stopped, removed, and any process is killed. The function
          does not return until cleanup is complete.
        * The returned :class:`ExecutionResult` contains only safe
          metadata and truncated output. No absolute host paths,
          source code, or environment values are included.
        """


class _NullSandbox(Sandbox):
    """A no-op sandbox for tests and as a safe default.

    This backend does not execute code. It validates requests against
    the configuration and returns a synthetic result. It is useful
    for unit tests that do not need real isolation and for callers
    that want a "sandbox that is clearly not real" placeholder.

    Use :class:`DockerSandbox` in production.
    """

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        # Validate language.
        if not self._config.is_language_supported(request.language):
            from core.sandbox.errors import UnsupportedLanguageError
            raise UnsupportedLanguageError(request.language)

        # Validate code.
        if request.code is None or not request.code:
            from core.sandbox.errors import SandboxValidationError
            raise SandboxValidationError("code must be a non-empty string")

        encoded = request.code.encode("utf-8", errors="strict")
        if len(encoded) > self._config.limits.max_code_bytes:
            from core.sandbox.errors import SandboxValidationError
            raise SandboxValidationError(
                f"code is too large (>{self._config.limits.max_code_bytes} bytes)"
            )

        # Validate timeout.
        if request.timeout_seconds <= 0:
            from core.sandbox.errors import SandboxValidationError
            raise SandboxValidationError("timeout_seconds must be positive")

        # We have nothing to actually execute. Return a synthetic result
        # that does not pretend the program ran.
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="sandbox backend is a no-op",
            timed_out=False,
            duration_seconds=0.0,
            output_truncated=False,
            execution_id=request.execution_id,
            language=request.language,
        )
