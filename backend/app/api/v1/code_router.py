"""Code execution-related routes for the v1 API.

Exposes endpoints for sandboxed code execution.
All operations go through the execute_code tool and policy engine.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Path
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from core.agent.types import ToolCall
from core.sandbox.interface import Sandbox
from core.sandbox.tools import EXECUTE_CODE_TOOL, _code_tool_callable, register_code_tools, CodeToolExecutor
from core.security.policy_engine import get_policy_engine
from core.tools.registry import DefaultToolRegistry

logger = logging.getLogger("sovereign-ai.api.code")

router = APIRouter(prefix="/code", tags=["code"])


def _iso_utc(ts: float) -> str:
    """Format a UTC POSIX timestamp as an ISO-8601 string for the frontend."""
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class ExecuteRequest(BaseModel):
    code: str = Field(..., description="The source code to execute. Must be a valid Python program.")
    language: Optional[str] = Field(None, description="Programming language to execute. Currently only 'python' is supported.")
    timeout: Optional[float] = Field(None, description="Maximum execution time in seconds.")
    stdin: Optional[str] = Field(None, description="Optional string to pass as stdin to the program.")


class ExecuteResponse(BaseModel):
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    output_truncated: bool
    execution_id: str
    language: str


class CodeExecutionResult(BaseModel):
    execution_id: str
    success: bool
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0
    output_truncated: bool = False


class CodeRunSummary(BaseModel):
    id: str
    title: str
    status: str  # "queued" | "running" | "succeeded" | "failed" | "timed_out"
    startedAt: str
    durationMs: Optional[int] = None
    exitCode: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class StoredCodeRun:
    def __init__(
        self,
        execution_id: str,
        title: str,
        code: str,
        success: bool,
        exit_code: Optional[int],
        stdout: str,
        stderr: str,
        timed_out: bool,
        duration_seconds: float,
        output_truncated: bool,
        started_at: float,
    ) -> None:
        self.execution_id = execution_id
        self.title = title
        self.code = code
        self.success = success
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.duration_seconds = duration_seconds
        self.output_truncated = output_truncated
        self.started_at = started_at


class CodeStore:
    def __init__(self) -> None:
        self._runs: dict[str, StoredCodeRun] = {}
        self._lock = asyncio.Lock()

    async def add(self, run: StoredCodeRun) -> None:
        async with self._lock:
            self._runs[run.execution_id] = run

    async def get(self, execution_id: str) -> Optional[StoredCodeRun]:
        async with self._lock:
            return self._runs.get(execution_id)

    async def list(self) -> List[StoredCodeRun]:
        async with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)


code_store = CodeStore()


def get_sandbox() -> Sandbox:
    """Get a sandbox instance for code execution."""
    from core.sandbox.docker_sandbox import DockerSandbox
    return DockerSandbox()


def get_tool_registry() -> DefaultToolRegistry:
    """Get a tool registry with the execute_code tool registered."""
    registry = DefaultToolRegistry()
    sandbox = get_sandbox()
    register_code_tools(registry, sandbox)
    return registry


@router.get("", response_model=List[CodeRunSummary])
async def list_code_runs() -> List[CodeRunSummary]:
    """List all executed code runs."""
    runs = await code_store.list()
    summaries: List[CodeRunSummary] = []
    for r in runs:
        if r.timed_out:
            st = "timed_out"
        elif r.success:
            st = "succeeded"
        else:
            st = "failed"
        summaries.append(
            CodeRunSummary(
                id=r.execution_id,
                title=r.title or f"Python run {r.execution_id[:8]}",
                status=st,
                startedAt=_iso_utc(r.started_at),
                durationMs=int(r.duration_seconds * 1000),
                exitCode=r.exit_code,
                stdout=r.stdout,
                stderr=r.stderr,
            )
        )
    return summaries


@router.post("/execute", response_model=ExecuteResponse)
async def execute_code(
    request: ExecuteRequest,
) -> ExecuteResponse:
    """Execute code in an isolated sandbox."""
    started_at = time.time()
    policy_engine = get_policy_engine()

    # Create a tool call for execute_code
    tool_call = ToolCall(
        call_id="execute_code_call",
        tool_name="execute_code",
        arguments={
            "code": request.code,
            "language": request.language or "python",
            "timeout": request.timeout,
            "stdin": request.stdin,
        },
    )

    # Check policy - code execution is DENY by default
    if policy_engine is not None:
        policy_decision = policy_engine.evaluate(tool_call)
        if not policy_decision.is_allowed():
            logger.warning("Policy denied execute_code: %s", policy_decision.reason)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=policy_decision.reason,
            )

    sandbox = get_sandbox()
    try:
        result = await _code_tool_callable(CodeToolExecutor(sandbox))(tool_call.arguments)
        resp = ExecuteResponse(**result)
        first_line = request.code.strip().splitlines()[0][:60] if request.code.strip() else "Python code"
        stored = StoredCodeRun(
            execution_id=resp.execution_id,
            title=first_line,
            code=request.code,
            success=resp.success,
            exit_code=resp.exit_code,
            stdout=resp.stdout,
            stderr=resp.stderr,
            timed_out=resp.timed_out,
            duration_seconds=resp.duration_seconds,
            output_truncated=resp.output_truncated,
            started_at=started_at,
        )
        await code_store.add(stored)
        return resp
    except Exception as exc:
        logger.exception("Error executing code: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Code execution failed",
        )


@router.get("/{execution_id}", response_model=CodeExecutionResult)
async def get_execution_result(
    execution_id: str = Path(..., description="The execution ID to retrieve"),
) -> CodeExecutionResult:
    """Retrieve the result of a previously executed code execution."""
    stored = await code_store.get(execution_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution result {execution_id} not found",
        )
    return CodeExecutionResult(
        execution_id=stored.execution_id,
        success=stored.success,
        exit_code=stored.exit_code,
        stdout=stored.stdout,
        stderr=stored.stderr,
        timed_out=stored.timed_out,
        duration_seconds=stored.duration_seconds,
        output_truncated=stored.output_truncated,
    )