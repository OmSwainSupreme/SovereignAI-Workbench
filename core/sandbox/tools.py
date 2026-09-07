"""Sandbox code-execution tool (Phase 5D).

This module registers an ``execute_code`` :class:`ToolDefinition` into a
:class:`DefaultToolRegistry`. The tool wraps a :class:`Sandbox`
instance (typically a :class:`DockerSandbox`) and exposes it to the
Agent as a callable tool.

Agent flow
----------

    Agent → ToolRegistry → execute_code tool → Sandbox → ExecutionResult

The Agent never calls the sandbox directly; it calls the tool, which
calls the sandbox.

Usage::

    from core.agent.registry import DefaultToolRegistry
    from core.sandbox.tools import register_code_tools

    registry = DefaultToolRegistry()
    sandbox = DockerSandbox()  # or _NullSandbox() for tests
    register_code_tools(registry, sandbox)

    # The agent can now request execute_code tool calls.

Validation
----------

The tool validates its inputs before calling the sandbox:

* ``language`` must be a supported language.
* ``code`` must be a non-empty string within the configured size limit.
* ``timeout`` must be a positive number within the configured maximum.

If any of these fail, the tool returns a :class:`ToolResult` with
``error=True`` and a safe error message. The sandbox is NOT called.

The sandbox itself also validates the request; the tool layer validation
is an additional defence.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Mapping, Optional

from core.agent.errors import ToolExecutionError
from core.agent.interfaces import ToolDefinition, ToolExecutor
from core.agent.registry import DefaultToolRegistry
from core.agent.types import ToolCall, ToolResult
from core.sandbox.errors import (
    SandboxBackendError,
    SandboxConfigError,
    SandboxError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
    UnsupportedLanguageError,
)
from core.sandbox.interface import Sandbox
from core.sandbox.types import ExecutionRequest, ExecutionResult


_logger = logging.getLogger("sovereign-ai.sandbox.tools")

# Maximum number of characters shown in a tool description excerpt.
_DESCRIPTION_EXCERPT_CHARS = 120


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

EXECUTE_CODE_TOOL = ToolDefinition(
    name="execute_code",
    description=(
        "Execute Python code inside an isolated sandbox. "
        "The sandbox has no network access, no host filesystem access, "
        "and runs with bounded CPU, memory, and execution time. "
        "The code runs as a non-root user in a disposable container. "
        "Returns stdout, stderr, exit code, and timing information."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "description": (
                    "Programming language to execute. "
                    "Currently only 'python' is supported."
                ),
                "default": "python",
            },
            "code": {
                "type": "string",
                "description": (
                    "The source code to execute. Must be a valid Python "
                    "program. The sandbox enforces a maximum code size."
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Maximum execution time in seconds. "
                    "Defaults to the sandbox's configured maximum."
                ),
                "minimum": 0.01,
            },
            "stdin": {
                "type": "string",
                "description": (
                    "Optional string to pass as stdin to the program."
                ),
            },
        },
        "required": ["code"],
    },
    output_description=(
        "A {success, exit_code, stdout, stderr, timed_out, "
        "duration_seconds, output_truncated} object."
    ),
    capability="code_execution",
)


# ---------------------------------------------------------------------------
# CodeToolExecutor
# ---------------------------------------------------------------------------


class CodeToolExecutor(ToolExecutor):
    """A :class:`ToolExecutor` that runs the ``execute_code`` tool.

    The executor is bound to a single :class:`Sandbox` instance. It
    translates :class:`ToolCall` arguments into :class:`ExecutionRequest`,
    calls the sandbox, and packages the :class:`ExecutionResult` into a
    :class:`ToolResult`.

    The executor is safe for concurrent use at the instance level; it
    holds only a reference to the sandbox, which is thread-safe.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            request = self._build_request(call.arguments, call.call_id)
        except (ValueError, TypeError) as exc:
            # Raised by __post_init__ of ExecutionRequest.
            _logger.info(
                "execute_code.validation  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message=f"Invalid request: {type(exc).__name__}",
                latency_ms=0.0,
            )
        except SandboxValidationError as exc:
            _logger.info(
                "execute_code.validation  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message=f"Validation error: {exc}",
                latency_ms=0.0,
            )

        try:
            result: ExecutionResult = await self._sandbox.execute(request)
        except SandboxTimeoutError:
            _logger.info("execute_code.timeout  call_id=%s", call.call_id)
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message="Execution timed out",
                latency_ms=0.0,
            )
        except SandboxConfigError as exc:
            _logger.warning(
                "execute_code.config_error  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message="Sandbox configuration error",
                latency_ms=0.0,
            )
        except SandboxBackendError as exc:
            _logger.warning(
                "execute_code.backend_error  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message="Sandbox backend error",
                latency_ms=0.0,
            )
        except SandboxExecutionError as exc:
            _logger.warning(
                "execute_code.execution_error  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message="Code execution failed",
                latency_ms=0.0,
            )
        except SandboxError as exc:
            _logger.warning(
                "execute_code.sandbox_error  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message="Sandbox error",
                latency_ms=0.0,
            )
        except Exception as exc:  # pragma: no cover — defensive
            _logger.warning(
                "execute_code.unexpected  call_id=%s  error=%s",
                call.call_id,
                type(exc).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message=f"Unexpected error: {type(exc).__name__}",
                latency_ms=0.0,
            )

        # The sandbox returned a result. Package it into a ToolResult.
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=result.to_dict(),
            error=False,
            error_message="",
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------ Request building

    def _build_request(
        self,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> ExecutionRequest:
        language = str(arguments.get("language", "python"))
        code = arguments.get("code")
        timeout_arg = arguments.get("timeout")
        stdin_arg = arguments.get("stdin")

        if not isinstance(code, str) or not code:
            raise SandboxValidationError("code must be a non-empty string")

        # Resolve timeout.
        timeout_seconds: float = self._sandbox.config.limits.timeout_seconds
        if timeout_arg is not None:
            try:
                timeout_seconds = float(timeout_arg)
            except (TypeError, ValueError):
                raise SandboxValidationError("timeout must be a number")

        # Resolve stdin.
        stdin: Optional[str] = None
        if stdin_arg is not None:
            if not isinstance(stdin_arg, str):
                stdin = str(stdin_arg)
            else:
                stdin = stdin_arg

        # Generate a unique execution ID if not provided.
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"

        return ExecutionRequest(
            execution_id=execution_id,
            language=language,
            code=code,
            timeout_seconds=timeout_seconds,
            stdin=stdin,
            resource_limits={},
        )


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


def register_code_tools(
    registry: DefaultToolRegistry,
    sandbox: Sandbox,
) -> None:
    """Register the ``execute_code`` tool into ``registry``.

    The tool's callable is bound to the given ``sandbox``.
    After this call, the registry can be handed to a
    :class:`core.agent.registry.SyncToolExecutor` and used by the agent.

    Args:
        registry: A :class:`DefaultToolRegistry` to populate.
        sandbox: The :class:`Sandbox` instance to use for execution.

    Raises:
        TypeError: if ``registry`` is not a :class:`DefaultToolRegistry`.
    """
    if not isinstance(registry, DefaultToolRegistry):
        raise TypeError(
            "register_code_tools requires a DefaultToolRegistry "
            "(other registries do not support callables)"
        )

    executor = CodeToolExecutor(sandbox)
    registry.register(EXECUTE_CODE_TOOL, _code_tool_callable(executor))

    _logger.info(
        "sandbox.tools.registered  tool=%s  backend=%s",
        EXECUTE_CODE_TOOL.name,
        sandbox.backend_name,
    )


# ---------------------------------------------------------------------------
# Callable bridge
# ---------------------------------------------------------------------------


def _code_tool_callable(executor: CodeToolExecutor):
    """Bridge from the registry's callable signature to the async executor."""

    async def _call(args: Mapping) -> object:
        from core.agent.types import ToolCall, ToolResult

        call = ToolCall(
            call_id="execute_code_call",
            tool_name=EXECUTE_CODE_TOOL.name,
            arguments=dict(args),
        )
        result: ToolResult = await executor.execute(call)
        if result.error:
            raise RuntimeError(result.error_message)
        return result.output

    return _call
