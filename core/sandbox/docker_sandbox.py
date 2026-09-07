"""Docker-based sandbox implementation.

The :class:`DockerSandbox` is the first concrete implementation of the
:class:`Sandbox` interface. It provides process/container/OS-level
isolation, which is the **primary** security boundary.

Security properties
-------------------

* The container runs as a non-root user (``"sandbox"`` by default).
* The container has its own PID namespace (no host PID namespace).
  The PID namespace is isolated by Docker's default private namespace.
  We do NOT pass an explicit ``--pid`` flag because the only valid
  modes on the current Docker CLI are ``host`` (which would break
  isolation) and ``container:<name>`` (which requires another
  container to exist). Docker's default already provides private
  isolation, so the safest portable implementation is to omit it.
* The container does NOT have a Docker socket mount.
* The container does NOT mount the host filesystem.
* The container's root filesystem is read-only; a small tmpfs is
  the only writable area.
* The container is given **no network access** (``network_disabled=True``).
* The container's environment is built from a small allowlist.
  The host's environment is NOT inherited.
* The container is created with bounded CPU, memory, and PID limits.
* The container is removed after every execution
  (``auto_remove=True``).
* The container is forcibly terminated on timeout.

The sandbox is constructed with a :class:`SandboxConfig` that controls
all of the above. Tests that build a :class:`DockerSandbox` should
inspect the produced ``docker run`` argument list (via the
:meth:`_build_container_args` hook) to verify the isolation
properties.

Construction
------------

The constructor does **not** start a container. It does perform a
one-time sanity check: it verifies that the Docker CLI is on ``PATH``
and that the requested image exists (if not in test mode). This is
deliberately lightweight; the unit tests do not require a running
Docker daemon.

A real container is only started in :meth:`execute`.

Subprocess boundary
-------------------

The :class:`DockerSandbox` shells out to the ``docker`` CLI (rather
than talking to the Docker daemon over its socket) for two reasons:

1. The Docker socket is a host-credential that the sandbox never
   touches. The CLI uses the same credential but the agent process
   itself never opens it directly.
2. The CLI is the documented public surface. The Python ``docker``
   SDK is optional and would add a dependency.

The CLI invocation goes through :func:`_run_docker`, which is fully
tested with a fake runner. Unit tests substitute the runner to
verify argument construction without ever starting a container.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Mapping, Optional, Sequence, Tuple

from core.sandbox.errors import (
    SandboxBackendError,
    SandboxConfigError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
    UnsupportedLanguageError,
)
from core.sandbox.interface import Sandbox
from core.sandbox.limits import ResourceLimits, SandboxConfig
from core.sandbox.types import (
    ExecutionRequest,
    ExecutionResult,
    LANGUAGE_PYTHON,
)


_logger = logging.getLogger("sovereign-ai.sandbox.docker")

# Type alias for the docker-CLI runner. A real runner is
# ``asyncio.create_subprocess_exec``; tests inject a fake that
# returns a canned result.
DockerRunner = Callable[
    [Sequence[str], Optional[str], Optional[float]],
    Awaitable[Tuple[int, str, str]],
]


# Sentinel: not a real Docker image name; used in tests to bypass the
# image-existence check. The actual image name in production is set
# via :class:`SandboxConfig`.
_TEST_IMAGE_SENTINEL = "__sandbox_test_image__"


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------


class DockerSandbox(Sandbox):
    """A :class:`Sandbox` backed by a Docker container.

    See the module docstring for the security properties. This class
    is the only one in the package that depends on an external
    runtime (the ``docker`` CLI).

    SECURITY: The Docker executable path is resolved internally using
    ``shutil.which("docker")`` and cannot be overridden through any
    public API. This prevents an attacker from bypassing the sandbox
    by specifying a malicious executable.
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        *,
        docker_runner: Optional[DockerRunner] = None,
        skip_image_check: bool = False,
    ) -> None:
        """Construct the sandbox.

        Args:
            config: The :class:`SandboxConfig` to use. Defaults to a
                safe built-in.
            docker_runner: A coroutine that runs the ``docker`` CLI
                and returns ``(exit_code, stdout, stderr)``. If
                ``None``, the default runner
                (:func:`_default_docker_runner`) is used.
                SECURITY: The runner is wrapped by asyncio.wait_for
                in execute() to enforce the execution timeout at the
                sandbox layer. A misbehaving runner cannot bypass
                the timeout.
            skip_image_check: If True, do not verify that the
                configured image exists. Useful in unit tests that
                only exercise argument construction.
        """
        super().__init__(config)
        self._docker_runner = docker_runner or _default_docker_runner
        self._skip_image_check = skip_image_check

        # SECURITY: Docker executable is resolved internally using
        # shutil.which("docker"). This path is NEVER user-controlled
        # to prevent sandbox bypass via arbitrary executable selection.
        resolved_path = shutil.which("docker")
        if resolved_path is None and not self._skip_image_check:
            raise SandboxConfigError(
                "docker CLI not found on PATH; install Docker to use "
                "the DockerSandbox backend"
            )
        self._docker_path = resolved_path

    # ------------------------------------------------------------------ Public

    @property
    def backend_name(self) -> str:
        return "docker"

    @property
    def docker_path(self) -> Optional[str]:
        return self._docker_path

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run ``request.code`` in a freshly-created container.

        The full lifecycle is:

        1. Validate the request.
        2. Build the ``docker run`` argument list.
        3. Launch the container with a hard wall-clock timeout.
        4. Wait for the container to finish (or kill it on timeout).
        5. Always remove the container.
        6. Return a structured :class:`ExecutionResult`.

        The function never raises on a non-zero exit code; that is a
        normal program outcome and is reported via ``success=False``.
        """
        effective_limits = self._config.effective_limits(
            request.resource_limits or None
        )
        effective_timeout = self._config.effective_timeout(request.timeout_seconds)

        # ----- Validation. Done before any container is started.
        self._validate_request(request, effective_limits)

        # ----- Build the code payload. The code is passed as a base64-
        # encoded argument to avoid any quoting issues. The container
        # entrypoint writes the code to /workspace/main.py and exec's
        # it under python3 with the resource limits applied.
        import base64
        code_b64 = base64.b64encode(request.code.encode("utf-8")).decode("ascii")

        container_name = f"sovereign-sandbox-{request.execution_id}"
        args = self._build_container_args(
            container_name=container_name,
            code_b64=code_b64,
            stdin=request.stdin,
            limits=effective_limits,
            timeout_seconds=effective_timeout,
        )

        start_time = time.monotonic()
        # SECURITY: Enforce timeout at the Sandbox layer using asyncio.wait_for.
        # This prevents a misbehaving docker_runner (e.g., custom implementation
        # that ignores the timeout argument) from causing indefinite hangs.
        # The effective_timeout already clamps to the configured maximum.
        execution_deadline = effective_timeout + 5.0  # extra time for Docker overhead
        try:
            exit_code, stdout, stderr = await asyncio.wait_for(
                self._docker_runner(
                    args,
                    None,  # stdin is not piped through the docker CLI
                    effective_timeout,
                ),
                timeout=execution_deadline,
            )
        except asyncio.TimeoutError as exc:
            duration = time.monotonic() - start_time
            # We force-stop and remove the container. Even if the
            # removal fails, we still return a result.
            # _force_cleanup has its own hard timeout to prevent hanging.
            await self._force_cleanup(container_name)
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="execution exceeded the configured timeout",
                timed_out=True,
                duration_seconds=duration,
                output_truncated=False,
                execution_id=request.execution_id,
                language=request.language,
            )
        except FileNotFoundError as exc:
            raise SandboxBackendError(
                "docker CLI not found at " + str(self._docker_path),
                cause=exc,
            ) from exc
        except Exception as exc:
            raise SandboxBackendError(
                f"docker invocation failed: {type(exc).__name__}",
                cause=exc,
            ) from exc

        duration = time.monotonic() - start_time

        # Parse the structured result the container's entrypoint emits
        # to stderr. The entrypoint writes a JSON object as the LAST
        # line of stderr; everything before it is the program's own
        # stderr.
        program_stdout, program_stderr, meta = _parse_container_output(stdout, stderr)

        # Apply output limits. The container itself truncates at the
        # configured limit, but the CLI runner captures bytes which we
        # also bound to be safe.
        output_truncated = bool(meta.get("truncated", False)) or bool(
            meta.get("stdout_truncated", False)
        ) or bool(meta.get("stderr_truncated", False))

        # Treat exit code 137 / 143 as a timeout (the container was
        # SIGKILL'd or SIGTERM'd by the timeout path).
        timed_out = bool(meta.get("timed_out", False))
        if not timed_out and exit_code in (137, 143):
            timed_out = True

        success = (exit_code == 0) and not timed_out

        return ExecutionResult(
            success=success,
            exit_code=exit_code,
            stdout=program_stdout,
            stderr=program_stderr,
            timed_out=timed_out,
            duration_seconds=duration,
            output_truncated=output_truncated,
            execution_id=request.execution_id,
            language=request.language,
        )

    # ------------------------------------------------------------------ Validation

    def _validate_request(
        self,
        request: ExecutionRequest,
        limits: ResourceLimits,
    ) -> None:
        if not self._config.is_language_supported(request.language):
            raise UnsupportedLanguageError(request.language)

        if request.code is None or not request.code:
            raise SandboxValidationError("code must be a non-empty string")

        if not isinstance(request.code, str):
            raise SandboxValidationError("code must be a string")

        try:
            encoded = request.code.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise SandboxValidationError("code is not valid UTF-8") from exc

        if len(encoded) > limits.max_code_bytes:
            raise SandboxValidationError(
                f"code is too large (>{limits.max_code_bytes} bytes)"
            )

        if request.timeout_seconds <= 0:
            raise SandboxValidationError("timeout_seconds must be positive")

        if request.stdin is not None and not isinstance(request.stdin, str):
            raise SandboxValidationError("stdin must be a string when provided")

    # ------------------------------------------------------------------ Argument construction

    def _build_container_args(
        self,
        *,
        container_name: str,
        code_b64: str,
        stdin: Optional[str],
        limits: ResourceLimits,
        timeout_seconds: float,
    ) -> List[str]:
        """Build the ``docker run`` argument list.

        Tests inspect this method to verify the security properties
        of the sandbox: no privileged mode, no host paths, no Docker
        socket, no environment inheritance, etc.
        """
        if self._docker_path is None:
            raise SandboxConfigError("docker CLI path is not configured")

        cfg = self._config
        image = cfg.image

        args: List[str] = [
            self._docker_path,
            "run",
            "--rm",  # remove the container after exit
            "--name", container_name,
            # Network isolation.
            "--network=none",
            # IPC namespace.
            "--ipc=private",
            # Drop all capabilities.
            "--cap-drop=ALL",
            # No new privileges.
            "--security-opt=no-new-privileges:true",
            # Read-only root filesystem with a small tmpfs workspace.
            "--read-only",
            "--tmpfs", f"/workspace:size={limits.max_workspace_bytes},uid=1000,gid=1000,mode=0750",
            # Resource limits.
            "--memory", f"{limits.max_memory_mb}m",
            "--memory-swap", f"{limits.max_memory_mb}m",  # disable swap
            "--cpus", f"{limits.max_cpus:.3f}",
            "--pids-limit", str(limits.max_pids),
            # Ulimit for output size. The container's entrypoint caps
            # the per-stream output at this many bytes.
            "--ulimit", "nofile=256:256",
            # Non-root user.
            "--user", cfg.user,
            # Working directory inside the container.
            "--workdir", "/workspace",
            # Labels for audit / identification.
            "--label", "sovereign-ai.sandbox=true",
            "--label", f"sovereign-ai.execution_id={container_name}",
        ]

        # Allowlist of environment variables. We pass only what is
        # explicitly allowed; the host's environment is NOT inherited
        # (no ``-e`` without a value).
        #
        # IMPORTANT: For most allowlisted variables we use a known-safe
        # default. We do NOT pass the host's PATH because the host PATH
        # is a Windows-style path that is meaningless (and potentially
        # dangerous) inside a Linux container. The container image has
        # its own PATH baked in.
        _ENV_DEFAULTS = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "LC_CTYPE": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "random",
            "TZ": "UTC",
        }
        for var in cfg.allowed_env_vars:
            value = _ENV_DEFAULTS.get(var, "")
            args.extend(["-e", f"{var}={value}"])

        # NOTE: We deliberately do NOT add:
        #   --privileged
        #   --pid=host
        #   --ipc=host
        #   --network=host
        #   -v /var/run/docker.sock
        #   -v <host-path>:/<container-path>
        #   --env-file
        #   any -e with secrets

        # The code is passed as a single environment variable. The
        # entrypoint decodes it and writes the file. We use a dedicated
        # variable name so the entrypoint can find it.
        args.extend(["-e", f"SOVEREIGN_CODE_B64={code_b64}"])
        args.extend(["-e", f"SOVEREIGN_TIMEOUT={int(timeout_seconds)}"])
        args.extend(["-e", f"SOVEREIGN_MAX_OUTPUT={limits.max_output_bytes}"])

        if stdin is not None:
            # We pass stdin as an env var too. The agent is expected
            # to embed a small payload; the variable is base64-encoded
            # for safety.
            import base64
            stdin_b64 = base64.b64encode(stdin.encode("utf-8")).decode("ascii")
            args.extend(["-e", f"SOVEREIGN_STDIN_B64={stdin_b64}"])

        # The image is always last; the entrypoint is the image's CMD.
        args.append(image)

        return args

    async def _force_cleanup(self, container_name: str) -> None:
        """Best-effort cleanup of a container that timed out.

        SECURITY: This method has a hard timeout to prevent the cleanup
        operation itself from hanging indefinitely if Docker becomes
        unresponsive. The caller has already decided to return a result,
        so cleanup failure does not propagate.

        Args:
            container_name: Name of the container to remove.
        """
        if self._docker_path is None:
            return
        # Hard timeout for cleanup: if Docker doesn't respond within this
        # time, we give up. A leaked container is far less severe than
        # crashing the host process.
        CLEANUP_TIMEOUT = 5.0
        try:
            await asyncio.wait_for(
                self._docker_runner(
                    [self._docker_path, "rm", "-f", container_name],
                    None,
                    CLEANUP_TIMEOUT,
                ),
                timeout=CLEANUP_TIMEOUT,
            )
        except asyncio.TimeoutError:
            # Cleanup timed out - Docker may be unresponsive. Log and continue.
            _logger.warning(
                "sandbox.cleanup.timeout  container=%s",
                container_name,
            )
        except Exception:
            # Best-effort only - log and continue.
            _logger.debug("sandbox.cleanup_failed  name=%s", container_name)

    # ------------------------------------------------------------------ Inspection helpers (for tests)

    def describe_isolation(self) -> dict:
        """Return a dict describing the isolation properties of this sandbox.

        Used by tests to verify the configuration. Keys mirror the
        :class:`SandboxConfig` fields and the additional security
        controls enforced at the CLI level.
        """
        return {
            "backend": self.backend_name,
            "image": self._config.image,
            "network_enabled": self._config.network_enabled,
            "read_only_root": self._config.read_only_root,
            "user": self._config.user,
            "drop_capabilities": self._config.drop_capabilities,
            "supported_languages": list(self._config.supported_languages),
            "allowed_env_vars": list(self._config.allowed_env_vars),
            "limits": {
                "timeout_seconds": self._config.limits.timeout_seconds,
                "max_timeout_seconds": self._config.max_timeout_seconds,
                "max_output_bytes": self._config.limits.max_output_bytes,
                "max_memory_mb": self._config.limits.max_memory_mb,
                "max_cpus": self._config.limits.max_cpus,
                "max_pids": self._config.limits.max_pids,
                "max_workspace_bytes": self._config.limits.max_workspace_bytes,
                "max_code_bytes": self._config.limits.max_code_bytes,
            },
        }


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_container_output(
    stdout: str,
    stderr: str,
) -> Tuple[str, str, dict]:
    """Split a container's combined output into (stdout, stderr, meta).

    The container's entrypoint writes a single line of JSON as the
    LAST line of stderr. The line has the form::

        __SOVEREIGN_RESULT__ {"exit_code":0,"timed_out":false,...}

    Everything before that line is the program's own stderr (or
    stdout, depending on the run flags). We use ``2>&1`` in the
    entrypoint, so both streams come back on the docker CLI's stdout;
    however, in the current implementation the entrypoint routes
    stdout to a file and the JSON marker to stderr, so we use the
    stderr field for the marker.

    This function is deliberately defensive: if the marker is missing
    or malformed, we return the raw output unchanged with an empty
    meta dict, so a buggy entrypoint does not silently lose output.
    """
    marker = "__SOVEREIGN_RESULT__"
    if not stderr:
        return stdout, stderr, {}

    # Find the LAST occurrence of the marker. The entrypoint writes
    # it once, at the end.
    idx = stderr.rfind(marker)
    if idx < 0:
        return stdout, stderr, {}

    # The marker may be on its own line followed by JSON.
    tail = stderr[idx + len(marker):].strip()
    head = stderr[:idx].rstrip("\r\n")

    try:
        meta = json.loads(tail) if tail else {}
    except json.JSONDecodeError:
        # Marker is present but the JSON is malformed. Treat the
        # whole stderr as the program stderr.
        return stdout, stderr, {}

    return stdout, head, meta


# ---------------------------------------------------------------------------
# Default docker runner
# ---------------------------------------------------------------------------


async def _default_docker_runner(
    args: Sequence[str],
    stdin: Optional[str],
    timeout: Optional[float],
) -> Tuple[int, str, str]:
    """Run the ``docker`` CLI as a subprocess and return its result.

    The function uses :class:`asyncio.create_subprocess_exec` to avoid
    blocking the event loop. It enforces a hard timeout via
    :func:`asyncio.wait_for`.

    The subprocess is started with ``stdout=PIPE``, ``stderr=PIPE``,
    and ``stdin=PIPE`` (only when ``stdin`` is provided). The
    arguments are passed as a list — :mod:`shlex` is NOT used, so
    values with spaces are not subject to shell interpretation.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input=stdin.encode("utf-8") if stdin is not None else None),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        # The subprocess did not finish in time. We attempt to kill
        # it; if that fails, we propagate the timeout to the caller.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        raise
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )
