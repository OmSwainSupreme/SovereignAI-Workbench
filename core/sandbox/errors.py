"""Exception hierarchy for the Secure Code Sandbox.

All sandbox errors are instances of :class:`SandboxError` (the common
base). This makes it easy for callers to catch "any sandbox failure"
while still being able to distinguish between categories:

* :class:`SandboxValidationError` — the request was rejected before
  execution. Examples: unsupported language, empty code, oversized code,
  invalid timeout, invalid resource request.
* :class:`SandboxConfigError` — the sandbox itself is misconfigured
  (e.g. image missing, runtime not installed).
* :class:`SandboxBackendError` — the underlying isolation backend
  (e.g. Docker daemon) returned an error.
* :class:`SandboxExecutionError` — the program ran but failed in a way
  that is not a simple non-zero exit code (e.g. could not be started).
* :class:`SandboxTimeoutError` — the execution exceeded its timeout.
* :class:`UnsupportedLanguageError` — convenience subclass for the
  common case of an unsupported language being requested.

Error messages do NOT include the source code, the absolute host paths
of the sandbox workspace, environment values, or other sensitive
information.
"""
from __future__ import annotations

from typing import Optional


class SandboxError(Exception):
    """Base class for all sandbox errors."""


class SandboxValidationError(SandboxError):
    """The :class:`ExecutionRequest` was rejected before execution.

    The request did not satisfy one of the validation requirements
    (e.g. unsupported language, empty code, oversized code, invalid
    timeout, disallowed resource override).
    """


class UnsupportedLanguageError(SandboxValidationError):
    """The requested language is not supported by this sandbox.

    Currently only ``"python"`` is supported.
    """

    def __init__(self, language: str) -> None:
        self.language = language
        super().__init__(f"Language not supported: {language!r}")


class SandboxConfigError(SandboxError):
    """The sandbox is misconfigured.

    Examples: the configured image does not exist, the container runtime
    is not available, the resource limits are internally inconsistent.
    """


class SandboxBackendError(SandboxError):
    """The underlying isolation backend (e.g. Docker) failed.

    Examples: the Docker daemon is not reachable, a low-level API call
    returned an error, the container could not be created.

    The underlying exception is preserved on :attr:`cause`.
    """

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        self.cause = cause
        super().__init__(message)


class SandboxExecutionError(SandboxError):
    """The program could not be executed in the sandbox.

    The sandbox itself is healthy, but the program could not be started
    (e.g. the binary is missing in the image, or the working directory
    could not be set up). The program itself never ran.

    Contrast with a non-zero exit code, which is a normal execution
    outcome and is reported as :class:`ExecutionResult` with
    ``success=False``, NOT as an exception.
    """


class SandboxTimeoutError(SandboxError):
    """The execution exceeded its timeout.

    Timeouts are also reported on :class:`ExecutionResult` as
    ``timed_out=True``; this exception is raised only when the timeout
    cannot be cleanly converted to a result (e.g. the backend could not
    be queried for the exit code).
    """
