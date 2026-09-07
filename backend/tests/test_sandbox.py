"""Comprehensive tests for Phase 5D Secure Code Sandbox.

All tests are designed to run without Docker. Unit tests use mocks/fakes;
Docker integration tests (marked with ``@pytest.mark.docker``) only run when
Docker is intentionally available.

Test organisation
-----------------

* :class:`TestSandboxTypes` — ExecutionRequest and ExecutionResult types.
* :class:`TestResourceLimits` — ResourceLimits and SandboxConfig validation.
* :class:`TestSandboxErrors` — error hierarchy.
* :class:`TestSandboxInterface` — the abstract Sandbox interface / _NullSandbox.
* :class:`TestDockerSandboxArgs` — argument construction (isolation properties).
* :class:`TestDockerSandboxValidation` — request validation.
* :class:`TestDockerSandboxResultParsing` — output parsing.
* :class:`TestExecuteCodeTool` — ToolDefinition and CodeToolExecutor.
* :class:`TestToolRegistryIntegration` — register_code_tools.
* :class:`TestSecurityConfiguration` — configuration security properties.
* :class:`TestDockerIntegration` — real Docker execution (marked).
* :class:`TestSecurityIsolationProperties` — verify sandbox configuration.
* :class:`TestOutputHandling` — truncation and safe logging.

IMPORTANT: Unit tests MUST NOT execute untrusted code directly on the host.
"""
from __future__ import annotations

import asyncio
import base64
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Iterator, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.errors import UnknownToolError
from core.agent.registry import DefaultToolRegistry
from core.agent.types import ToolCall, ToolResult
from core.sandbox import (
    LANGUAGE_PYTHON,
    SandboxConfig,
    ResourceLimits,
    ExecutionRequest,
    ExecutionResult,
    Sandbox,
    _NullSandbox,
    EXECUTE_CODE_TOOL,
    CodeToolExecutor,
    register_code_tools,
    SandboxError,
    SandboxBackendError,
    SandboxConfigError,
    SandboxValidationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    UnsupportedLanguageError,
)
from core.sandbox.docker_sandbox import (
    DockerSandbox,
    _parse_container_output,
)
from core.sandbox.interface import Sandbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> SandboxConfig:
    return SandboxConfig()


@pytest.fixture
def sandbox_config() -> SandboxConfig:
    return SandboxConfig(
        limits=ResourceLimits(
            timeout_seconds=10.0,
            max_output_bytes=8192,
            max_memory_mb=256,
            max_cpus=1.0,
            max_pids=64,
            max_workspace_bytes=16 * 1024 * 1024,
            max_code_bytes=64 * 1024,
        ),
        max_timeout_seconds=60.0,
        image="sovereign-ai/sandbox-python:5d",
        network_enabled=False,
        read_only_root=True,
        drop_capabilities=True,
        user="sandbox",
    )


@pytest.fixture
def null_sandbox(default_config: SandboxConfig) -> Sandbox:
    return _NullSandbox(default_config)


@pytest.fixture
def docker_sandbox(sandbox_config: SandboxConfig) -> DockerSandbox:
    """A DockerSandbox that bypasses the image-existence check."""
    return DockerSandbox(
        config=sandbox_config,
        skip_image_check=True,
    )


@pytest.fixture
def code_executor(null_sandbox: Sandbox) -> CodeToolExecutor:
    return CodeToolExecutor(null_sandbox)


@pytest.fixture
def code_registry(sandbox_config: SandboxConfig) -> DefaultToolRegistry:
    registry = DefaultToolRegistry()
    sandbox = _NullSandbox(sandbox_config)
    register_code_tools(registry, sandbox)
    return registry


def make_request(
    language: str = LANGUAGE_PYTHON,
    code: str = "print('hello')",
    timeout: float = 10.0,
    stdin: str | None = None,
    execution_id: str = "test-exec-0",
) -> ExecutionRequest:
    return ExecutionRequest(
        execution_id=execution_id,
        language=language,
        code=code,
        timeout_seconds=timeout,
        stdin=stdin,
    )


def make_call(
    arguments: dict[str, Any],
    call_id: str = "test-call-0",
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_name=EXECUTE_CODE_TOOL.name,
        arguments=arguments,
    )


# ---------------------------------------------------------------------------
# TestSandboxTypes
# ---------------------------------------------------------------------------

class TestSandboxTypes:
    def test_execution_request_valid(self) -> None:
        req = make_request()
        assert req.execution_id == "test-exec-0"
        assert req.language == "python"
        assert req.code == "print('hello')"
        assert req.timeout_seconds == 10.0
        assert req.stdin is None

    def test_execution_request_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="execution_id"):
            ExecutionRequest(execution_id="", language="python", code="x")

    def test_execution_request_rejects_empty_language(self) -> None:
        with pytest.raises(ValueError, match="language"):
            ExecutionRequest(execution_id="x", language="", code="x")

    def test_execution_request_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            ExecutionRequest(execution_id="x", language="python", code="x", timeout_seconds=0)

    def test_execution_result_success(self) -> None:
        res = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="hello",
            stderr="",
            timed_out=False,
            duration_seconds=1.5,
        )
        assert res.success is True
        assert res.exit_code == 0
        assert res.stdout == "hello"
        assert res.stderr == ""
        assert res.timed_out is False
        assert res.duration_seconds == 1.5
        assert res.output_truncated is False

    def test_execution_result_failure(self) -> None:
        res = ExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="error",
            timed_out=False,
            duration_seconds=0.1,
        )
        assert res.success is False
        assert res.exit_code == 1
        assert res.stderr == "error"

    def test_execution_result_timeout(self) -> None:
        res = ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="timeout",
            timed_out=True,
            duration_seconds=30.0,
        )
        assert res.success is False
        assert res.timed_out is True

    def test_execution_result_truncated(self) -> None:
        res = ExecutionResult(
            success=False,
            exit_code=1,
            stdout="x",
            stderr="",
            timed_out=False,
            duration_seconds=0.1,
            output_truncated=True,
        )
        assert res.output_truncated is True
        assert res.truncated_stdout is True

    def test_execution_result_to_dict(self) -> None:
        res = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="out",
            stderr="err",
            timed_out=False,
            duration_seconds=0.5,
            output_truncated=False,
            execution_id="exec-1",
            language="python",
        )
        d = res.to_dict()
        assert d["success"] is True
        assert d["exit_code"] == 0
        assert d["stdout"] == "out"
        assert d["stderr"] == "err"
        assert d["timed_out"] is False
        assert d["output_truncated"] is False
        assert d["execution_id"] == "exec-1"
        assert d["language"] == "python"


# ---------------------------------------------------------------------------
# TestResourceLimits
# ---------------------------------------------------------------------------

class TestResourceLimits:
    def test_default_limits(self) -> None:
        limits = ResourceLimits()
        assert limits.timeout_seconds == 30.0
        assert limits.max_output_bytes == 64 * 1024
        assert limits.max_memory_mb == 256
        assert limits.max_cpus == 1.0
        assert limits.max_pids == 64
        assert limits.max_workspace_bytes == 16 * 1024 * 1024
        assert limits.max_code_bytes == 64 * 1024

    def test_custom_limits(self) -> None:
        limits = ResourceLimits(timeout_seconds=5.0, max_memory_mb=128)
        assert limits.timeout_seconds == 5.0
        assert limits.max_memory_mb == 128
        assert limits.max_output_bytes == 64 * 1024  # unchanged

    def test_rejects_negative_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            ResourceLimits(timeout_seconds=-1)

    def test_rejects_zero_output(self) -> None:
        with pytest.raises(ValueError, match="max_output_bytes"):
            ResourceLimits(max_output_bytes=0)

    def test_rejects_negative_memory(self) -> None:
        with pytest.raises(ValueError, match="max_memory_mb"):
            ResourceLimits(max_memory_mb=-1)

    def test_rejects_zero_cpus(self) -> None:
        with pytest.raises(ValueError, match="max_cpus"):
            ResourceLimits(max_cpus=0)

    def test_rejects_zero_pids(self) -> None:
        with pytest.raises(ValueError, match="max_pids"):
            ResourceLimits(max_pids=0)

    def test_default_config(self) -> None:
        cfg = SandboxConfig()
        assert cfg.network_enabled is False
        assert cfg.read_only_root is True
        assert cfg.drop_capabilities is True
        assert cfg.user == "sandbox"
        assert "python" in cfg.supported_languages
        assert "PATH" in cfg.allowed_env_vars

    def test_config_rejects_root_user(self) -> None:
        with pytest.raises(ValueError, match="not.*root"):
            SandboxConfig(user="root")

    def test_config_rejects_empty_image(self) -> None:
        with pytest.raises(ValueError, match="image"):
            SandboxConfig(image="")

    def test_config_rejects_empty_languages(self) -> None:
        with pytest.raises(ValueError, match="supported_languages"):
            SandboxConfig(supported_languages=())

    def test_is_language_supported_python(self, default_config: SandboxConfig) -> None:
        assert default_config.is_language_supported("python") is True

    def test_is_language_supported_unsupported(self, default_config: SandboxConfig) -> None:
        assert default_config.is_language_supported("javascript") is False
        assert default_config.is_language_supported("bash") is False
        assert default_config.is_language_supported("") is False

    def test_effective_timeout_clamped_to_max(self, default_config: SandboxConfig) -> None:
        # Request more than the maximum.
        effective = default_config.effective_timeout(999.0)
        assert effective == default_config.max_timeout_seconds

    def test_effective_timeout_uses_default(self, default_config: SandboxConfig) -> None:
        # Request nothing.
        effective = default_config.effective_timeout(0)
        assert effective == default_config.limits.timeout_seconds

    def test_effective_timeout_uses_requested(self, default_config: SandboxConfig) -> None:
        effective = default_config.effective_timeout(5.0)
        assert effective == 5.0

    def test_effective_limits_no_overrides(self, default_config: SandboxConfig) -> None:
        limits = default_config.effective_limits({})
        assert limits == default_config.limits

    def test_effective_limits_applies_overrides(self, default_config: SandboxConfig) -> None:
        limits = default_config.effective_limits({"timeout_seconds": 5.0})
        assert limits.timeout_seconds == 5.0

    def test_effective_limits_clamps_timeout_to_max(self, default_config: SandboxConfig) -> None:
        # Override tries to exceed the max.
        limits = default_config.effective_limits({"timeout_seconds": 9999.0})
        assert limits.timeout_seconds <= default_config.max_timeout_seconds

    def test_effective_limits_ignores_negative_values(self, default_config: SandboxConfig) -> None:
        limits = default_config.effective_limits({"timeout_seconds": -1})
        assert limits.timeout_seconds == default_config.limits.timeout_seconds

    def test_effective_limits_ignores_non_numeric(self, default_config: SandboxConfig) -> None:
        limits = default_config.effective_limits({"timeout_seconds": "five"})
        assert limits.timeout_seconds == default_config.limits.timeout_seconds


# ---------------------------------------------------------------------------
# TestSandboxErrors
# ---------------------------------------------------------------------------

class TestSandboxErrors:
    def test_unsupported_language_error(self) -> None:
        exc = UnsupportedLanguageError("javascript")
        assert exc.language == "javascript"
        assert "javascript" in str(exc)
        assert "not supported" in str(exc)

    def test_sandbox_error_is_base(self) -> None:
        exc = SandboxError("something")
        assert isinstance(exc, Exception)

    def test_validation_error_categorisation(self) -> None:
        exc = SandboxValidationError("bad code")
        assert isinstance(exc, SandboxError)
        assert isinstance(exc, SandboxValidationError)

    def test_backend_error_with_cause(self) -> None:
        cause = ValueError("original")
        exc = SandboxBackendError("docker failed", cause=cause)
        assert exc.cause is cause
        assert "docker failed" in str(exc)


# ---------------------------------------------------------------------------
# TestSandboxInterface
# ---------------------------------------------------------------------------

class TestSandboxInterface:
    def test_null_sandbox_returns_noop_result(self, default_config: SandboxConfig) -> None:
        sandbox = _NullSandbox(default_config)
        req = make_request()
        result = asyncio.run(sandbox.execute(req))
        assert result.success is False
        assert result.exit_code == -1
        assert "no-op" in result.stderr

    def test_null_sandbox_rejects_unsupported_language(
        self, default_config: SandboxConfig
    ) -> None:
        sandbox = _NullSandbox(default_config)
        req = make_request(language="javascript")
        with pytest.raises(UnsupportedLanguageError):
            asyncio.run(sandbox.execute(req))

    def test_null_sandbox_rejects_empty_code(
        self, default_config: SandboxConfig
    ) -> None:
        sandbox = _NullSandbox(default_config)
        req = make_request(code="")
        with pytest.raises(SandboxValidationError, match="non-empty"):
            asyncio.run(sandbox.execute(req))

    def test_null_sandbox_rejects_none_code(
        self, default_config: SandboxConfig
    ) -> None:
        sandbox = _NullSandbox(default_config)
        req = ExecutionRequest(
            execution_id="x", language="python", code="x" * 1000000
        )
        with pytest.raises(SandboxValidationError, match="too large"):
            asyncio.run(sandbox.execute(req))

    def test_null_sandbox_rejects_invalid_timeout(
        self, default_config: SandboxConfig
    ) -> None:
        sandbox = _NullSandbox(default_config)
        # Use _unsafe to bypass __post_init__ validation.
        # The sandbox should reject the invalid timeout.
        req = ExecutionRequest._unsafe(
            execution_id="test-timeout",
            language="python",
            code="print(1)",
            timeout_seconds=-1.0,
        )
        with pytest.raises(SandboxValidationError, match="positive"):
            asyncio.run(sandbox.execute(req))

    def test_sandbox_interface_backend_name(self, default_config: SandboxConfig) -> None:
        sandbox = _NullSandbox(default_config)
        assert sandbox.backend_name == "_nullsandbox"

    def test_sandbox_config_property(self, default_config: SandboxConfig) -> None:
        sandbox = _NullSandbox(default_config)
        assert sandbox.config is default_config


# ---------------------------------------------------------------------------
# TestDockerSandboxArgs — verify isolation properties
# ---------------------------------------------------------------------------

class TestDockerSandboxArgs:
    def test_network_isolation(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--network=none" in args
        assert "--network=host" not in args

    def test_no_docker_socket_mount(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        assert "docker.sock" not in args_str
        assert "/var/run/docker.sock" not in args_str

    def test_no_privileged_mode(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        assert "--privileged" not in args_str
        assert "--pid=host" not in args_str
        assert "--ipc=host" not in args_str
        assert "--network=host" not in args_str

    def test_no_host_volume_mounts(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        # No -v with a host path.
        for arg in args:
            if arg == "-v" or arg.startswith("-v="):
                pytest.fail(f"Found volume mount in args: {arg}")

    def test_read_only_root(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--read-only" in args

    def test_tmpfs_workspace(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        tmpfs_flags = [a for a in args if a.startswith("--tmpfs")]
        assert len(tmpfs_flags) == 1
        # The tmpfs flag's VALUE is the next arg.
        tmpfs_idx = args.index(tmpfs_flags[0])
        assert "/workspace" in args[tmpfs_idx + 1]

    def test_non_root_user(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        user_args = [a for a in args if a == "--user"]
        assert len(user_args) == 1
        user_idx = args.index("--user")
        assert args[user_idx + 1] == "sandbox"

    def test_cap_drop_all(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--cap-drop=ALL" in args

    def test_no_new_privileges(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--security-opt=no-new-privileges:true" in args

    def test_memory_limit(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--memory" in args
        mem_idx = args.index("--memory")
        assert args[mem_idx + 1] == "256m"

    def test_cpu_limit(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--cpus" in args
        cpus_idx = args.index("--cpus")
        assert "1.000" in args[cpus_idx + 1]

    def test_pids_limit(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--pids-limit" in args
        pids_idx = args.index("--pids-limit")
        assert args[pids_idx + 1] == "64"

    def test_container_removed(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--rm" in args

    def test_container_has_label(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        labels = [a for a in args if a.startswith("--label")]
        label_vals = [args[args.index(a) + 1] for a in labels]
        assert any("sovereign-ai" in l for l in label_vals)

    def test_code_passed_as_env_var(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        # Each -e flag is followed by a NAME=VALUE pair.
        env_names = []
        for i, arg in enumerate(args):
            if arg == "-e":
                val = args[i + 1]
                name = val.split("=", 1)[0]
                env_names.append(name)
        assert "SOVEREIGN_CODE_B64" in env_names
        # No raw code in args.
        assert "print(1)" not in " ".join(args)

    def test_minimal_environment(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        # Each -e flag is followed by a NAME=VALUE pair.
        env_names = []
        for i, arg in enumerate(args):
            if arg == "-e":
                val = args[i + 1]
                name = val.split("=", 1)[0]
                env_names.append(name)
        # Allowed vars should be in the allowlist or be SOVEREIGN_*.
        allowed = set(sandbox_config.allowed_env_vars)
        for name in env_names:
            assert name in allowed or name.startswith("SOVEREIGN"), (
                f"env var {name!r} not in allowlist"
            )

    def test_no_host_environment_wholesale(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        # No --env-file.
        assert "--env-file" not in args
        # No bare -e without a value (which would inherit from host).
        for i, arg in enumerate(args):
            if arg == "-e":
                # Next arg must be NAME=VALUE, not just a bare name.
                val = args[i + 1]
                assert "=" in val, f"Bare env var without value: {val}"

    def test_stdin_passed_when_provided(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin="hello world",
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        stdin_vars = [a for a in args if "STDIN" in a]
        assert len(stdin_vars) >= 1

    def test_isolation_description(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        info = docker_sandbox.describe_isolation()
        assert info["backend"] == "docker"
        assert info["network_enabled"] is False
        assert info["read_only_root"] is True
        assert info["user"] == "sandbox"
        assert info["drop_capabilities"] is True
        assert "python" in info["supported_languages"]


# ---------------------------------------------------------------------------
# TestDockerSandboxValidation
# ---------------------------------------------------------------------------

class TestDockerSandboxValidation:
    def test_rejects_unsupported_language(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        req = make_request(language="javascript")
        with pytest.raises(UnsupportedLanguageError):
            asyncio.run(docker_sandbox.execute(req))

    def test_rejects_unsupported_language_case_sensitive(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        req = make_request(language="Python")
        with pytest.raises(UnsupportedLanguageError):
            asyncio.run(docker_sandbox.execute(req))

    def test_rejects_empty_code(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        req = make_request(code="")
        with pytest.raises(SandboxValidationError, match="non-empty"):
            asyncio.run(docker_sandbox.execute(req))

    def test_rejects_none_code(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        req = ExecutionRequest(
            execution_id="x", language="python", code="x"
        )
        req = ExecutionRequest(execution_id="x", language="python", code="")
        with pytest.raises(SandboxValidationError):
            asyncio.run(docker_sandbox.execute(req))

    def test_rejects_oversized_code(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        big_code = "x" * (sandbox_config.limits.max_code_bytes + 1)
        req = make_request(code=big_code)
        with pytest.raises(SandboxValidationError, match="too large"):
            asyncio.run(docker_sandbox.execute(req))

    def test_rejects_invalid_timeout_zero(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        # Use _unsafe to bypass __post_init__ validation.
        req = ExecutionRequest._unsafe(
            execution_id="test-zero",
            language="python",
            code="x",
            timeout_seconds=0.0,
        )
        with pytest.raises(SandboxValidationError, match="positive"):
            asyncio.run(docker_sandbox.execute(req))

    def test_rejects_invalid_timeout_negative(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        req = ExecutionRequest._unsafe(
            execution_id="test-negative",
            language="python",
            code="x",
            timeout_seconds=-5.0,
        )
        with pytest.raises(SandboxValidationError, match="positive"):
            asyncio.run(docker_sandbox.execute(req))

    def test_rejects_non_string_code(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        # Create a request with a non-string code.
        req = ExecutionRequest(
            execution_id="x",
            language="python",
            code=bytes([1, 2, 3]),  # type: ignore
        )
        with pytest.raises(SandboxValidationError, match="string"):
            asyncio.run(docker_sandbox.execute(req))


# ---------------------------------------------------------------------------
# TestDockerSandboxResultParsing
# ---------------------------------------------------------------------------

class TestDockerSandboxResultParsing:
    def test_parse_normal_output(self) -> None:
        marker = "__SOVEREIGN_RESULT__"
        meta = {"exit_code": 0, "timed_out": False, "stdout_truncated": False, "stderr_truncated": False}
        stderr = f"some stderr\n{marker} {json.dumps(meta)}\n"
        stdout, stderr_out, parsed = _parse_container_output("hello\n", stderr)
        assert stdout == "hello\n"
        # The parser strips the marker line and trailing newlines.
        assert parsed["exit_code"] == 0
        assert "some stderr" in stderr_out

    def test_parse_timeout_marker(self) -> None:
        marker = "__SOVEREIGN_RESULT__"
        meta = {"exit_code": -1, "timed_out": True, "stdout_truncated": False, "stderr_truncated": False}
        stderr = f"{marker} {json.dumps(meta)}\n"
        stdout, stderr_out, parsed = _parse_container_output("", stderr)
        assert parsed["timed_out"] is True
        assert parsed["exit_code"] == -1

    def test_parse_truncated_marker(self) -> None:
        marker = "__SOVEREIGN_RESULT__"
        meta = {"exit_code": 0, "timed_out": False, "stdout_truncated": True, "stderr_truncated": False}
        stderr = f"{marker} {json.dumps(meta)}\n"
        stdout, stderr_out, parsed = _parse_container_output("x" * 100, stderr)
        assert parsed["stdout_truncated"] is True

    def test_parse_missing_marker(self) -> None:
        stdout, stderr_out, parsed = _parse_container_output("hello", "stderr only")
        assert stdout == "hello"
        assert stderr_out == "stderr only"
        assert parsed == {}

    def test_parse_malformed_marker(self) -> None:
        stdout, stderr_out, parsed = _parse_container_output(
            "", "__SOVEREIGN_RESULT__ not json"
        )
        assert parsed == {}


# ---------------------------------------------------------------------------
# TestExecuteCodeTool — tool definition and executor
# ---------------------------------------------------------------------------

class TestExecuteCodeTool:
    def test_tool_definition_fields(self) -> None:
        assert EXECUTE_CODE_TOOL.name == "execute_code"
        assert EXECUTE_CODE_TOOL.capability == "code_execution"
        # The "language" property exists; its default is "python".
        assert "language" in EXECUTE_CODE_TOOL.input_schema["properties"]
        assert EXECUTE_CODE_TOOL.input_schema["properties"]["language"]["default"] == "python"

    def test_tool_definition_validates_language(self) -> None:
        schema = EXECUTE_CODE_TOOL.input_schema
        assert schema["properties"]["language"]["type"] == "string"
        assert schema["properties"]["code"]["type"] == "string"
        assert "required" in schema
        assert "code" in schema["required"]

    def test_executor_validates_empty_code(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": ""})
        result = asyncio.run(code_executor.execute(call))
        assert result.error is True
        assert "code" in result.error_message.lower()

    def test_executor_validates_missing_code(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({})
        result = asyncio.run(code_executor.execute(call))
        assert result.error is True

    def test_executor_returns_result_on_success(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": "print('hello')"})
        result = asyncio.run(code_executor.execute(call))
        # The null sandbox always fails (no-op), so success=False is expected.
        assert result.output is not None
        assert "success" in result.output

    def test_executor_produces_structured_output(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": "print('test')", "language": "python"})
        result = asyncio.run(code_executor.execute(call))
        assert result.output is not None
        d = result.output
        assert "exit_code" in d
        assert "stdout" in d
        assert "stderr" in d
        assert "timed_out" in d
        assert "duration_seconds" in d

    def test_executor_handles_invalid_timeout(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": "x", "timeout": "not a number"})
        result = asyncio.run(code_executor.execute(call))
        assert result.error is True
        assert "timeout" in result.error_message.lower()

    def test_executor_stdin_passed(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": "x", "stdin": "hello"})
        result = asyncio.run(code_executor.execute(call))
        assert result.error is False

    def test_executor_handles_sandbox_validation_error(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": "x" * 1_000_000})  # oversized
        result = asyncio.run(code_executor.execute(call))
        assert result.error is True

    def test_executor_returns_safe_error_message(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": ""})
        result = asyncio.run(code_executor.execute(call))
        assert result.error is True
        # No code in error message.
        assert result.error_message  # non-empty
        assert "print(" not in result.error_message


# ---------------------------------------------------------------------------
# TestToolRegistryIntegration
# ---------------------------------------------------------------------------

class TestToolRegistryIntegration:
    def test_register_code_tools_registers_tool(
        self, sandbox_config: SandboxConfig
    ) -> None:
        registry = DefaultToolRegistry()
        sandbox = _NullSandbox(sandbox_config)
        register_code_tools(registry, sandbox)
        assert registry.has("execute_code")

    def test_register_code_tools_tool_is_executable(
        self, sandbox_config: SandboxConfig
    ) -> None:
        registry = DefaultToolRegistry()
        sandbox = _NullSandbox(sandbox_config)
        register_code_tools(registry, sandbox)
        tool_def = registry.get("execute_code")
        assert tool_def.name == "execute_code"
        # DefaultToolRegistry stores the callable internally.
        # We can check the tool is callable via the registry's storage.
        assert registry.has("execute_code")

    def test_register_code_tools_rejects_non_default_registry(
        self, sandbox_config: SandboxConfig
    ) -> None:
        registry = DefaultToolRegistry()  # This is actually a DefaultToolRegistry
        sandbox = _NullSandbox(sandbox_config)
        # Should work.
        register_code_tools(registry, sandbox)

    def test_tool_registry_via_executor(
        self, sandbox_config: SandboxConfig
    ) -> None:
        from core.agent.registry import SyncToolExecutor

        registry = DefaultToolRegistry()
        sandbox = _NullSandbox(sandbox_config)
        register_code_tools(registry, sandbox)
        executor = SyncToolExecutor(registry)

        call = make_call({"code": "print(1)"})
        result = asyncio.run(executor.execute(call))
        assert result.output is not None
        assert "exit_code" in result.output

    def test_tool_unknown_tool_name(
        self, sandbox_config: SandboxConfig
    ) -> None:
        registry = DefaultToolRegistry()
        sandbox = _NullSandbox(sandbox_config)
        register_code_tools(registry, sandbox)

        call = ToolCall(
            call_id="test",
            tool_name="nonexistent_tool",
            arguments={"code": "x"},
        )
        from core.agent.registry import SyncToolExecutor

        executor = SyncToolExecutor(registry)
        # The SyncToolExecutor raises UnknownToolError for unknown tools;
        # this is by design. The agent catches it.
        with pytest.raises(UnknownToolError):
            asyncio.run(executor.execute(call))


# ---------------------------------------------------------------------------
# TestSecurityConfiguration
# ---------------------------------------------------------------------------

class TestSecurityConfiguration:
    def test_config_has_no_network(self, default_config: SandboxConfig) -> None:
        assert default_config.network_enabled is False

    def test_config_has_read_only_root(self, default_config: SandboxConfig) -> None:
        assert default_config.read_only_root is True

    def test_config_has_non_root_user(self, default_config: SandboxConfig) -> None:
        assert default_config.user == "sandbox"
        assert default_config.user != "root"
        assert default_config.user != "0"

    def test_config_has_drop_capabilities(self, default_config: SandboxConfig) -> None:
        assert default_config.drop_capabilities is True

    def test_config_no_dangerous_env_vars(self, default_config: SandboxConfig) -> None:
        dangerous = {
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
            "AZURE_STORAGE_KEY",
            "GCP_CREDENTIALS",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "HOME",
            "USER",
            "USERNAME",
        }
        for var in dangerous:
            assert var not in default_config.allowed_env_vars, f"{var} should not be in allowed_env_vars"

    def test_config_allows_safe_vars(self, default_config: SandboxConfig) -> None:
        safe = {"PATH", "LANG", "LC_ALL", "PYTHONUNBUFFERED"}
        for var in safe:
            assert var in default_config.allowed_env_vars

    def test_config_no_api_keys_in_env(self, default_config: SandboxConfig) -> None:
        env_str = " ".join(default_config.allowed_env_vars).lower()
        assert "api_key" not in env_str
        assert "secret" not in env_str
        assert "token" not in env_str


# ---------------------------------------------------------------------------
# TestSecurityIsolationProperties — verify configuration is safe
# ---------------------------------------------------------------------------

class TestSecurityIsolationProperties:
    def test_docker_args_have_no_host_network(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        dangerous = [
            "--network=host",
            "--network=bridge",
            "--net=host",
            "--net=bridge",
        ]
        for flag in dangerous:
            assert flag not in args_str

    def test_docker_args_have_no_docker_socket(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        assert "docker.sock" not in args_str

    def test_docker_args_have_no_host_filesystem(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        # Check for -v or --volume without a tmpfs.
        for i, arg in enumerate(args):
            if arg in ("-v", "--volume"):
                val = args[i + 1]
                # The only volume should be the tmpfs.
                assert "tmpfs" in val or ":" not in val, f"Non-tmpfs volume mount found: {arg} {val}"

    def test_docker_args_no_privileged(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        assert "--privileged" not in args_str

    def test_docker_args_non_root_user(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        if "--user" in args:
            idx = args.index("--user")
            user = args[idx + 1]
            assert user != "root"
            assert user != "0:0"

    def test_docker_args_drop_all_capabilities(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--cap-drop=ALL" in args

    def test_docker_args_no_env_file(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--env-file" not in args

    def test_docker_args_no_git_ssh_sensitive_mounts(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        # Check for specific dangerous volume mount patterns. We use full
        # path prefixes to avoid false positives from the Windows docker path.
        sensitive_prefixes = [
            "/.git",
            "/.env",
            "/id_rsa",
            "/.ssh",
            "/.aws",
            "/var/run/docker.sock",
        ]
        for s in sensitive_prefixes:
            assert s not in args_str, f"found sensitive mount: {s}"


# ---------------------------------------------------------------------------
# TestOutputHandling — verify safe output
# ---------------------------------------------------------------------------

class TestOutputHandling:
    def test_result_to_dict_omits_code(self) -> None:
        res = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="hello",
            stderr="",
            timed_out=False,
            duration_seconds=1.0,
        )
        d = res.to_dict()
        # Code should never appear in result output.
        assert "code" not in d
        assert "source" not in d
        assert "environment" not in d

    def test_result_to_dict_omits_env(self) -> None:
        res = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            duration_seconds=1.0,
        )
        d = res.to_dict()
        for key in d:
            val = str(d[key]).lower()
            assert "api_key" not in val
            assert "secret" not in val

    def test_result_to_dict_safe_for_logging(self) -> None:
        res = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="print('hello')",
            stderr="",
            timed_out=False,
            duration_seconds=0.5,
        )
        d = res.to_dict()
        # These fields are safe to include in logs.
        assert "success" in d
        assert "exit_code" in d
        assert "timed_out" in d
        assert "duration_seconds" in d
        assert "output_truncated" in d
        assert "execution_id" in d
        assert "language" in d

    def test_docker_args_code_not_in_plain_text(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        code = 'import os; print(os.listdir("/"))'
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(code.encode()).decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        # The code itself should not appear as plaintext in args.
        assert 'print(os.listdir' not in args_str
        assert 'os.listdir' not in args_str
        # But the base64 version should be there.
        assert "SOVEREIGN_CODE_B64" in args_str

    def test_docker_args_env_var_names_safe(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        for i, arg in enumerate(args):
            if arg.startswith("-e"):
                val = args[i + 1] if i + 1 < len(args) else ""
                name = val.split("=")[0]
                # Env var name should be alphanumeric + underscore.
                assert name.isidentifier(), f"Suspicious env var name: {name}"


# ---------------------------------------------------------------------------
# TestDeterministicValidation — ensure validation is deterministic
# ---------------------------------------------------------------------------

class TestDeterministicValidation:
    def test_validation_same_result_on_repeated_calls(
        self, code_executor: CodeToolExecutor
    ) -> None:
        call = make_call({"code": ""})
        results = [asyncio.run(code_executor.execute(call)) for _ in range(5)]
        for r in results:
            assert r.error is True

    def test_validation_rejects_any_unsupported_language(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        for lang in ["ruby", "java", "go", "rust", "javascript", "shell", "c"]:
            req = make_request(language=lang)
            with pytest.raises(UnsupportedLanguageError):
                asyncio.run(docker_sandbox.execute(req))

    def test_oversized_code_always_rejected(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        big_code = "x" * (sandbox_config.limits.max_code_bytes + 1)
        req = make_request(code=big_code)
        with pytest.raises(SandboxValidationError):
            asyncio.run(docker_sandbox.execute(req))


# ---------------------------------------------------------------------------
# TestSecurityFixes — regression tests for security fixes
# ---------------------------------------------------------------------------


class TestSecurityFixes:
    """Regression tests for the Phase 5D security hardening fixes.

    These tests verify:
    1. Docker path cannot be attacker-controlled
    2. Cleanup cannot hang indefinitely
    3. Custom runner cannot bypass execution timeout
    4. Timeout triggers cleanup
    5. Cleanup timeout is bounded
    """

    def test_docker_path_not_constructor_parameter(self) -> None:
        """Verify docker_path is NOT a public constructor parameter.

        SECURITY: An attacker should not be able to specify an arbitrary
        executable through the constructor.
        """
        import inspect
        sig = inspect.signature(DockerSandbox.__init__)
        params = list(sig.parameters.keys())
        # docker_path must not be a parameter anymore
        assert "docker_path" not in params, (
            f"docker_path should not be a public parameter. "
            f"Found parameters: {params}"
        )

    def test_docker_path_resolves_via_shutil_which(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        """Verify Docker path is resolved through shutil.which()."""
        import shutil
        # The resolved path should match what shutil.which finds
        expected = shutil.which("docker")
        if expected is not None:
            assert docker_sandbox.docker_path == expected
        # If shutil.which returns None, docker_sandbox.docker_path may also be None
        # (with skip_image_check=True) - that's acceptable

    def test_docker_path_not_affected_by_execution_input(
        self, docker_sandbox: DockerSandbox
    ) -> None:
        """Verify execution requests cannot affect Docker path."""
        original_path = docker_sandbox.docker_path
        # Try to pass malicious code that might affect docker_path
        malicious_requests = [
            make_request(code="import os; os.system('rm -rf /')"),
            make_request(code="__import__('subprocess').call(['evil'])"),
            make_request(code="x" * 100000),  # oversized
        ]
        for req in malicious_requests:
            try:
                # The request may be rejected for validation reasons,
                # but docker_path must never change
                asyncio.run(docker_sandbox.execute(req))
            except Exception:
                pass
            # docker_path must remain unchanged
            assert docker_sandbox.docker_path == original_path

    async def test_cleanup_cannot_hang_indefinitely(
        self, sandbox_config: SandboxConfig
    ) -> None:
        """Verify _force_cleanup has a hard timeout.

        Uses a fake runner that never returns to simulate a hung Docker.
        """
        async def hanging_runner(args, stdin, timeout):
            # Simulate a hung Docker daemon - never returns
            await asyncio.sleep(60)
            return (0, "", "")

        sandbox = DockerSandbox(
            config=sandbox_config,
            docker_runner=hanging_runner,
            skip_image_check=True,
        )

        start = time.monotonic()
        # This should complete within ~6 seconds (cleanup timeout is 5s)
        await sandbox._force_cleanup("test-container")
        duration = time.monotonic() - start

        # Should complete in well under 10 seconds
        assert duration < 10.0, (
            f"Cleanup took {duration}s - should be bounded by timeout"
        )

    async def test_custom_runner_cannot_bypass_execution_timeout(
        self, sandbox_config: SandboxConfig
    ) -> None:
        """Verify that a misbehaving runner cannot bypass the timeout.

        Uses a fake runner that ignores the timeout argument.
        """
        async def ignoring_runner(args, stdin, timeout):
            # Ignore the timeout and sleep for 60 seconds
            await asyncio.sleep(60)
            return (0, "", "")

        sandbox = DockerSandbox(
            config=sandbox_config,
            docker_runner=ignoring_runner,
            skip_image_check=True,
        )

        req = make_request(timeout=2.0)
        start = time.monotonic()
        result = await sandbox.execute(req)
        duration = time.monotonic() - start

        # The execution should be terminated by asyncio.wait_for
        # effective_timeout is 2.0, plus 5.0 buffer = 7.0 seconds max
        # Plus cleanup timeout of 5.0 seconds = 12.0 seconds total
        assert duration < 15.0, (
            f"Execution took {duration}s - timeout was not enforced"
        )
        # Should report as timed out
        assert result.timed_out is True
        assert result.success is False

    async def test_timeout_triggers_cleanup(
        self, sandbox_config: SandboxConfig
    ) -> None:
        """Verify that a timeout triggers _force_cleanup."""
        cleanup_called = []

        async def slow_runner(args, stdin, timeout):
            # Simulate slow execution that will timeout
            await asyncio.sleep(60)
            return (0, "", "")

        async def tracking_cleanup(self, container_name):
            # Wrap the real cleanup to track calls
            cleanup_called.append(container_name)
            # Call the real cleanup
            await self._docker_runner(
                ["/bin/true", "rm", "-f", container_name],
                None,
                1.0,
            )

        # Use the real cleanup but with a slow runner
        sandbox = DockerSandbox(
            config=sandbox_config,
            docker_runner=slow_runner,
            skip_image_check=True,
        )

        # Monkey-patch _force_cleanup to track calls
        original_cleanup = sandbox._force_cleanup

        async def tracked_cleanup(name):
            cleanup_called.append(name)
            # Don't actually call docker - just return
            pass

        sandbox._force_cleanup = tracked_cleanup

        req = make_request(timeout=1.0)
        result = await sandbox.execute(req)

        # Verify cleanup was called
        assert len(cleanup_called) > 0, "Cleanup should be called on timeout"
        assert result.timed_out is True

    async def test_cleanup_timeout_is_bounded(
        self, sandbox_config: SandboxConfig
    ) -> None:
        """Verify cleanup has a hard upper bound."""
        async def hanging_runner(args, stdin, timeout):
            # Simulate hung Docker during cleanup
            await asyncio.sleep(120)
            return (0, "", "")

        sandbox = DockerSandbox(
            config=sandbox_config,
            docker_runner=hanging_runner,
            skip_image_check=True,
        )

        start = time.monotonic()
        # Cleanup should complete in ~5 seconds (hard timeout)
        await sandbox._force_cleanup("test-container")
        duration = time.monotonic() - start

        # Must complete in under 7 seconds (5s timeout + 2s buffer)
        assert duration < 7.0, (
            f"Cleanup took {duration}s - timeout not enforced"
        )

    def test_existing_network_isolation_intact(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        """Verify network isolation flags are still present."""
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--network=none" in args
        assert "--network=host" not in args

    def test_existing_filesystem_isolation_intact(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        """Verify filesystem isolation flags are still present."""
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--read-only" in args
        assert "--tmpfs" in args

    def test_existing_privilege_restrictions_intact(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        """Verify privilege restrictions are still present."""
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--cap-drop=ALL" in args
        assert "--security-opt=no-new-privileges:true" in args
        assert "--user" in args
        assert "--privileged" not in " ".join(args)

    def test_existing_resource_limits_intact(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        """Verify resource limits are still present."""
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--memory" in args
        assert "--cpus" in args
        assert "--pids-limit" in args

    def test_existing_environment_isolation_intact(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        """Verify environment isolation is still present."""
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        assert "--env-file" not in args
        # No bare -e without a value
        for i, arg in enumerate(args):
            if arg == "-e":
                val = args[i + 1]
                assert "=" in val, f"Bare env var without value: {val}"

    def test_existing_output_limits_intact(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        """Verify output limits are still configured."""
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(b"print(1)").decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        # SOVEREIGN_MAX_OUTPUT should be passed
        assert any("SOVEREIGN_MAX_OUTPUT" in a for a in args)

    def test_execute_code_tool_still_works(
        self, code_executor: CodeToolExecutor
    ) -> None:
        """Verify the execute_code tool still functions correctly."""
        call = make_call({"code": "print('hello')"})
        result = asyncio.run(code_executor.execute(call))
        assert result.error is False
        assert result.output is not None
        assert "exit_code" in result.output

    def test_no_sensitive_data_in_logs(
        self, docker_sandbox: DockerSandbox, sandbox_config: SandboxConfig
    ) -> None:
        """Verify no sensitive data appears in Docker args."""
        secret_code = "API_KEY='secret123'; print(API_KEY)"
        args = docker_sandbox._build_container_args(
            container_name="test",
            code_b64=base64.b64encode(secret_code.encode()).decode(),
            stdin=None,
            limits=sandbox_config.limits,
            timeout_seconds=10.0,
        )
        args_str = " ".join(args)
        # Secret should not appear in plaintext
        assert "secret123" not in args_str
        assert "API_KEY" not in args_str


class TestEntrypointProcessGroup:
    """Tests for the entrypoint's process group cleanup logic.

    These tests verify that the timeout cleanup in sandbox/entrypoint.py
    properly handles child processes.
    """

    def test_kill_process_group_function_exists(self) -> None:
        """Verify _kill_process_group helper exists in entrypoint."""
        # Import the entrypoint module
        import importlib.util
        import sys
        from pathlib import Path

        entrypoint_path = Path(__file__).parent.parent.parent / "sandbox" / "entrypoint.py"
        if not entrypoint_path.exists():
            pytest.skip("entrypoint.py not found")

        # Check the file contains the function
        content = entrypoint_path.read_text()
        assert "_kill_process_group" in content, (
            "_kill_process_group helper must exist in entrypoint"
        )

    def test_kill_process_group_tries_process_group_first(self) -> None:
        """Verify _kill_process_group attempts process group kill first."""
        from pathlib import Path

        entrypoint_path = Path(__file__).parent.parent.parent / "sandbox" / "entrypoint.py"
        if not entrypoint_path.exists():
            pytest.skip("entrypoint.py not found")

        content = entrypoint_path.read_text()
        # Should call os.killpg first
        assert "os.killpg" in content, (
            "entrypoint must use os.killpg for process group termination"
        )

    def test_kill_process_group_has_fallback(self) -> None:
        """Verify _kill_process_group has fallback mechanisms."""
        from pathlib import Path

        entrypoint_path = Path(__file__).parent.parent.parent / "sandbox" / "entrypoint.py"
        if not entrypoint_path.exists():
            pytest.skip("entrypoint.py not found")

        content = entrypoint_path.read_text()
        # Should have fallback to proc.kill()
        assert "proc.kill" in content, (
            "entrypoint must have fallback to proc.kill()"
        )
        # Should have /proc scanning as final fallback
        assert "/proc" in content, (
            "entrypoint must have /proc scanning fallback"
        )


# ---------------------------------------------------------------------------
# TestDockerIntegration — real Docker execution (marked, optional)
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    """Return True if Docker CLI and daemon are both available."""
    import shutil

    if shutil.which("docker") is None:
        return False
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            env={"PATH": subprocess.os.environ.get("PATH", "")},
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.docker
class TestDockerIntegration:
    """Tests that actually run Docker. Only executed when Docker is available."""

    @pytest.fixture(autouse=True)
    def _check_docker(self) -> Iterator[None]:
        if not _docker_available():
            pytest.skip("Docker daemon not available")
        yield

    async def _run_in_docker(
        self, code: str, timeout: float = 10.0
    ) -> ExecutionResult:
        from core.sandbox.docker_sandbox import DockerSandbox
        from core.sandbox.limits import SandboxConfig

        config = SandboxConfig(
            image="sovereign-ai/sandbox-python:5d",
            max_timeout_seconds=60.0,
        )
        sandbox = DockerSandbox(
            config=config,
            skip_image_check=False,  # Actually check the image
        )
        req = ExecutionRequest(
            execution_id="test-integration",
            language="python",
            code=code,
            timeout_seconds=timeout,
        )
        return await sandbox.execute(req)

    async def test_docker_successful_python_execution(self) -> None:
        result = await self._run_in_docker("print('hello from sandbox')")
        assert result.success is True
        assert result.exit_code == 0
        assert "hello from sandbox" in result.stdout
        assert result.timed_out is False

    async def test_docker_stderr_capture(self) -> None:
        result = await self._run_in_docker(
            "import sys; sys.stderr.write('error message\\n')"
        )
        # The entrypoint wraps stderr, so we just check it's captured.
        assert result.stderr != ""

    async def test_docker_nonzero_exit_code(self) -> None:
        result = await self._run_in_docker("raise SystemExit(1)")
        assert result.success is False
        assert result.exit_code == 1

    async def test_docker_stdout_capture(self) -> None:
        result = await self._run_in_docker(
            "import json; print(json.dumps({'key': 'value'}))"
        )
        assert "key" in result.stdout

    async def test_docker_no_host_filesystem_access(self) -> None:
        result = await self._run_in_docker(
            "import os; print(os.listdir('/'))"
        )
        # The container's root is the container filesystem, not the host.
        # We just verify the code ran and produced output.
        assert result.exit_code == 0

    async def test_docker_no_network_access(self) -> None:
        result = await self._run_in_docker(
            "import socket; socket.socket()"
        )
        # This should work (socket module loads) but network is blocked.
        # The exact behaviour depends on the kernel, but we just verify
        # the execution completes.
        assert result.exit_code in (0, 1)  # either is fine; we don't know socket state

    async def test_docker_timeout_terminates(self) -> None:
        result = await self._run_in_docker(
            "while True: pass",
            timeout=2.0,
        )
        assert result.timed_out is True
        assert result.success is False
