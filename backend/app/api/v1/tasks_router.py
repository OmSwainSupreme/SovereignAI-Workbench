"""Task-related routes for the v1 API.

Exposes endpoints to submit agent tasks, list tasks, cancel tasks, and retrieve task results.
Supports real-time execution phase tracking, duplicate submission protection, and safe concurrency.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.services.model_service import get_model_service
from core.agent import Agent, AgentConfig, AgentResult, AgentStatus
from core.agent.errors import AgentExecutionError
from core.agent.interfaces import Planner, ToolExecutor, ToolRegistry, Verifier
from core.agent.types import (
    AgentMessage,
    AgentResult,
    AgentState,
    Artifact,
    ExecutionPhase,
    Observation,
    Plan,
    PlanStep,
    Source,
    ToolCall,
    ToolResult,
)
from core.agent.registry import DefaultToolRegistry
from core.agent.planner import SimplePlanner
from core.agent.verifier import SimpleVerifier
from core.agent.errors import (
    InvalidTaskError,
    IterationLimitExceededError,
    MaxToolCallsExceededError,
    PlanningFailureError,
    RepetitiveToolCallError,
    RoutingFailureError,
    ToolExecutionError,
    UnknownToolError,
    VerificationFailureError,
)
from core.routing import ModelRouter, NoSuitableModelError, RoutingError
from core.routing.registry import load_default_registry
from core.tools.workspace import Workspace, WorkspaceError
from core.tools.file_tools import (
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    create_file_tool_executor,
)
from core.tools.document_tools import CREATE_DOCUMENT_TOOL, create_document_tool_executor
from core.vision.tools import (
    OCR_IMAGE_TOOL,
    ANALYZE_IMAGE_TOOL,
    register_vision_tools,
)
from core.vision.image_loader import ImageLoader
from core.vision.ollama_ocr import OllamaOCRProvider
from core.vision.ollama_vision import OllamaVisionProvider
from core.security.policy_engine import get_policy_engine, PolicyDecision
from core.audit_logger import get_audit_logger
from core.agent.intent import classify_intent, TaskIntent
import inspect
import re

logger = logging.getLogger("sovereign-ai.api.tasks")


# Pydantic models for the frontend shapes
class PolicyNotice(BaseModel):
    id: str
    decision: str  # "allowed" | "denied" | "requires_approval"
    reason: str
    approvable: Optional[bool] = None


class KnowledgeSource(BaseModel):
    id: str
    title: str
    snippet: str
    score: Optional[float] = None
    collection: Optional[str] = None


class ChatMessage(BaseModel):
    id: str
    role: str  # "user" | "agent"
    content: str
    createdAt: str
    attachments: Optional[List[str]] = None
    sources: Optional[List[KnowledgeSource]] = None
    grounded: Optional[bool] = None
    pending: Optional[bool] = None
    thinking: Optional[str] = None


class ToolEvent(BaseModel):
    id: str
    label: str
    status: str  # "running" | "succeeded" | "failed"
    detail: Optional[str] = None


class TaskSummary(BaseModel):
    id: str
    title: str
    status: str  # "queued" | "running" | "completed" | "failed" | "cancelled" | "denied" | "awaiting_approval"
    updatedAt: str
    preview: Optional[str] = None
    phase: Optional[str] = None
    phaseLabel: Optional[str] = None
    model: Optional[str] = None
    elapsedSeconds: Optional[float] = None


class Task(TaskSummary):
    messages: List[ChatMessage]
    toolEvents: List[ToolEvent]
    policy: Optional[PolicyNotice] = None


def _iso_utc(ts: float) -> str:
    """Format a UTC POSIX timestamp as a millisecond-precision ISO-8601 string."""
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _preview_from_messages(messages: Sequence) -> Optional[str]:
    """Return a short preview snippet from the agent's most recent response."""
    for msg in reversed(messages):
        if getattr(msg, "role", None) in ("assistant", "agent"):
            content = getattr(msg, "content", None) or ""
            if content:
                return content[:200]
    return None


class TaskRecord:
    """Live internal record of a task and its execution state."""

    def __init__(
        self,
        task_id: str,
        task_description: str,
        attachments: List[str],
        result: AgentResult,
        created_at: float,
        completed_at: Optional[float] = None,
        status: str = "queued",
        phase: str = "queued",
        phase_label: str = "Queued",
        model: Optional[str] = None,
        tool_events: Optional[List[ToolEvent]] = None,
        error: Optional[str] = None,
        streaming_content: Optional[str] = None,
        streaming_thinking: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.task_description = task_description
        self.attachments = attachments
        self.result = result
        self.created_at = created_at
        self.completed_at = completed_at
        self.status = status
        self.phase = phase
        self.phase_label = phase_label
        self.model = model
        self.tool_events: List[ToolEvent] = tool_events or []
        self.error = error
        self.streaming_content: Optional[str] = streaming_content
        self.streaming_thinking: Optional[str] = streaming_thinking
        self.asyncio_task: Optional[asyncio.Task] = None

    def elapsed_seconds(self) -> float:
        end_time = self.completed_at if self.completed_at is not None else time.time()
        return round(max(0.0, end_time - self.created_at), 1)

    def to_summary(self) -> TaskSummary:
        ts = self.completed_at if self.completed_at is not None else self.created_at
        preview = _preview_from_messages(self.result.messages) if self.result else None
        return TaskSummary(
            id=self.task_id,
            title=self.task_description,
            status=self.status,
            updatedAt=_iso_utc(ts),
            preview=preview,
            phase=self.phase,
            phaseLabel=self.phase_label,
            model=self.model,
            elapsedSeconds=self.elapsed_seconds(),
        )

    def to_task(self) -> Task:
        summary = self.to_summary()
        messages: List[ChatMessage] = []
        if self.result and self.result.messages:
            for idx, msg in enumerate(self.result.messages):
                role = "user" if msg.role == "user" else "agent"
                msg_attachments = self.attachments if role == "user" and self.attachments else None
                clean_content = msg.content
                clean_thinking = getattr(msg, "thinking", None)
                if role == "agent":
                    if "</think>" in clean_content:
                        clean_content = re.sub(r"^.*?<\/think>", "", clean_content, flags=re.DOTALL).strip()
                    elif "<think>" in clean_content:
                        clean_content = re.sub(r"<think>.*?</think>", "", clean_content, flags=re.DOTALL).strip()
                    if not clean_thinking and self.streaming_thinking:
                        clean_thinking = self.streaming_thinking
                messages.append(
                    ChatMessage(
                        id=f"{self.task_id}-msg-{idx}",
                        role=role,
                        content=clean_content,
                        createdAt=summary.updatedAt,
                        attachments=msg_attachments,
                        thinking=clean_thinking,
                    )
                )
        else:
            messages.append(
                ChatMessage(
                    id=f"{self.task_id}-msg-0",
                    role="user",
                    content=self.task_description,
                    createdAt=_iso_utc(self.created_at),
                    attachments=self.attachments if self.attachments else None,
                )
            )
            if self.streaming_content or self.streaming_thinking:
                clean_stream = self.streaming_content or ""
                if "</think>" in clean_stream:
                    clean_stream = re.sub(r"^.*?<\/think>", "", clean_stream, flags=re.DOTALL).strip()
                elif "<think>" in clean_stream:
                    clean_stream = ""
                
                clean_thinking = self.streaming_thinking or ""
                if not clean_thinking and self.streaming_content and "<think>" in self.streaming_content:
                    think_match = re.search(r"<think>(.*?)(?:<\/think>|$)", self.streaming_content, flags=re.DOTALL)
                    if think_match:
                        clean_thinking = think_match.group(1).strip()

                if clean_stream or clean_thinking:
                    messages.append(
                        ChatMessage(
                            id=f"{self.task_id}-msg-1",
                            role="agent",
                            content=clean_stream,
                            createdAt=_iso_utc(time.time()),
                            pending=True,
                            thinking=clean_thinking or None,
                        )
                    )

        tool_events = list(self.tool_events)
        if self.result and self.result.tool_calls:
            obs_map = {obs.call_id: obs for obs in self.result.observations}
            for idx, tc in enumerate(self.result.tool_calls):
                existing = next(
                    (t for t in tool_events if t.id == tc.call_id or t.id == f"{self.task_id}-tool-{idx}"),
                    None,
                )
                if not existing:
                    obs = obs_map.get(tc.call_id)
                    content = obs.content if obs else ""
                    if content.startswith("[Tool error]"):
                        st = "failed"
                    elif content.startswith(("[Policy Denied]", "[Approval Required]")):
                        st = "failed"
                    else:
                        st = "succeeded"
                    tool_events.append(
                        ToolEvent(
                            id=tc.call_id or f"{self.task_id}-tool-{idx}",
                            label=tc.tool_name,
                            status=st,
                            detail=content,
                        )
                    )

        return Task(
            id=summary.id,
            title=summary.title,
            status=summary.status,
            updatedAt=summary.updatedAt,
            preview=summary.preview,
            phase=summary.phase,
            phaseLabel=summary.phaseLabel,
            model=summary.model,
            elapsedSeconds=summary.elapsedSeconds,
            messages=messages,
            toolEvents=tool_events,
            policy=None,
        )


class TaskStore:
    """In-memory store for agent task records with concurrency control and deduplication."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock: Optional[asyncio.Lock] = None
        self._exec_lock: Optional[asyncio.Lock] = None
        self._recent_submissions: dict[str, tuple[str, float]] = {}

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_exec_lock(self) -> asyncio.Lock:
        if self._exec_lock is None:
            self._exec_lock = asyncio.Lock()
        return self._exec_lock

    async def create_task_record(
        self,
        task_id: str,
        task_description: str,
        attachments: List[str],
        result: AgentResult,
        created_at: float,
        completed_at: Optional[float] = None,
        status: str = "queued",
        phase: str = "queued",
        phase_label: str = "Queued",
        model: Optional[str] = None,
    ) -> TaskRecord:
        async with self._get_lock():
            rec = TaskRecord(
                task_id=task_id,
                task_description=task_description,
                attachments=attachments,
                result=result,
                created_at=created_at,
                completed_at=completed_at,
                status=status,
                phase=phase,
                phase_label=phase_label,
                model=model,
            )
            self._tasks[task_id] = rec
            return rec

    async def create_task(
        self, task_id: str, result: AgentResult, created_at: float, completed_at: Optional[float] = None
    ) -> None:
        """Backward-compatible create_task."""
        async with self._get_lock():
            rec = self._tasks.get(task_id)
            if rec:
                rec.result = result
                rec.completed_at = completed_at
                rec.status = "completed" if result.status == AgentStatus.COMPLETE else "failed"
            else:
                self._tasks[task_id] = TaskRecord(
                    task_id=task_id,
                    task_description=result.task,
                    attachments=[],
                    result=result,
                    created_at=created_at,
                    completed_at=completed_at,
                    status="completed" if result.status == AgentStatus.COMPLETE else "failed",
                )

    async def get_task_record(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def update_streaming_state(
        self,
        task_id: str,
        *,
        streaming_content: Optional[str] = None,
        streaming_thinking: Optional[str] = None,
        phase: Optional[str] = None,
        phase_label: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Lock-free synchronous update for high-frequency token and phase events."""
        rec = self._tasks.get(task_id)
        if not rec:
            return
        if streaming_content is not None:
            rec.streaming_content = streaming_content
        if streaming_thinking is not None:
            rec.streaming_thinking = streaming_thinking
        if phase is not None:
            rec.phase = phase
        if phase_label is not None:
            rec.phase_label = phase_label
        if model is not None:
            rec.model = model

    def update_tool_event(self, task_id: str, tool_event: ToolEvent) -> None:
        """Synchronous tool event update for live tracking without async lock contention."""
        rec = self._tasks.get(task_id)
        if not rec:
            return
        existing_idx = next(
            (i for i, t in enumerate(rec.tool_events) if t.id == tool_event.id), -1
        )
        if existing_idx >= 0:
            rec.tool_events[existing_idx] = tool_event
        else:
            rec.tool_events.append(tool_event)

    async def get_task(self, task_id: str) -> Optional[tuple[AgentResult, float, Optional[float]]]:
        """Backward-compatible get_task."""
        async with self._get_lock():
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            return (rec.result, rec.created_at, rec.completed_at)

    async def list_tasks(self) -> List[tuple[str, AgentResult, float, Optional[float]]]:
        """Backward-compatible list_tasks."""
        async with self._get_lock():
            return [
                (rec.task_id, rec.result, rec.created_at, rec.completed_at)
                for rec in self._tasks.values()
            ]

    async def list_task_records(self) -> List[TaskRecord]:
        async with self._get_lock():
            return list(self._tasks.values())

    async def delete_task(self, task_id: str) -> None:
        async with self._get_lock():
            self._tasks.pop(task_id, None)

    async def check_duplicate(self, signature: str, window_seconds: float = 5.0) -> Optional[str]:
        """Return the existing active task_id if submitted within window or still active."""
        now = time.time()
        async with self._get_lock():
            if signature in self._recent_submissions:
                prev_id, prev_time = self._recent_submissions[signature]
                rec = self._tasks.get(prev_id)
                if (now - prev_time < window_seconds) or (rec and rec.status in ("queued", "running")):
                    return prev_id
            return None

    async def record_submission(self, signature: str, task_id: str) -> None:
        now = time.time()
        async with self._get_lock():
            self._recent_submissions[signature] = (task_id, now)

    async def update_task_state(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        phase_label: Optional[str] = None,
        model: Optional[str] = None,
        tool_event: Optional[ToolEvent] = None,
        result: Optional[AgentResult] = None,
        completed_at: Optional[float] = None,
        error: Optional[str] = None,
        streaming_content: Optional[str] = None,
        streaming_thinking: Optional[str] = None,
    ) -> None:
        async with self._get_lock():
            rec = self._tasks.get(task_id)
            if not rec:
                return
            if status is not None:
                rec.status = status
            if phase is not None:
                rec.phase = phase
            if phase_label is not None:
                rec.phase_label = phase_label
            if model is not None:
                rec.model = model
            if tool_event is not None:
                existing_idx = next(
                    (i for i, t in enumerate(rec.tool_events) if t.id == tool_event.id), -1
                )
                if existing_idx >= 0:
                    rec.tool_events[existing_idx] = tool_event
                else:
                    rec.tool_events.append(tool_event)
            if result is not None:
                rec.result = result
            if completed_at is not None:
                rec.completed_at = completed_at
            if error is not None:
                rec.error = error
            if streaming_content is not None:
                rec.streaming_content = streaming_content
            if streaming_thinking is not None:
                rec.streaming_thinking = streaming_thinking

    async def cancel_task(self, task_id: str) -> bool:
        async with self._get_lock():
            rec = self._tasks.get(task_id)
            if not rec:
                return False
            if rec.status in ("completed", "failed", "cancelled"):
                return False
            rec.status = "cancelled"
            rec.phase = "cancelled"
            rec.phase_label = "Cancelled by user"
            rec.completed_at = time.time()
            if rec.asyncio_task and not rec.asyncio_task.done():
                rec.asyncio_task.cancel()
            return True

    async def run_or_enqueue(
        self,
        task_id: str,
        task_description: str,
        attachments: List[str],
    ) -> None:
        exec_lock = self._get_exec_lock()
        rec = await self.get_task_record(task_id)
        if rec and rec.status == "cancelled":
            return

        if exec_lock.locked():
            await self.update_task_state(
                task_id,
                status="queued",
                phase="queued",
                phase_label="Queued behind current task...",
            )

        async with exec_lock:
            rec = await self.get_task_record(task_id)
            if rec and rec.status == "cancelled":
                return
            await self.update_task_state(
                task_id,
                status="running",
                phase="understanding",
                phase_label="🧠 Understanding request...",
            )
            rec = await self.get_task_record(task_id)
            if rec:
                rec.asyncio_task = asyncio.current_task()
            await run_agent_task(task_id, task_description, attachments)


task_store = TaskStore()


def get_workspace() -> Workspace:
    """Return a workspace instance for file operations."""
    workspace_path = settings.workspace_path or "workspace"
    return Workspace(workspace_path)


def get_tool_registry() -> ToolRegistry:
    """Build a tool registry with the core tools and vision tools."""
    registry = DefaultToolRegistry()
    registry.register(LIST_FILES_TOOL)
    registry.register(READ_FILE_TOOL)
    registry.register(WRITE_FILE_TOOL)
    registry.register(CREATE_DOCUMENT_TOOL)
    registry.register(OCR_IMAGE_TOOL)
    registry.register(ANALYZE_IMAGE_TOOL)
    return registry


def get_tool_executor(registry: ToolRegistry, workspace: Workspace) -> ToolExecutor:
    """Build a tool executor for the given registry and workspace."""
    file_executor = create_file_tool_executor(workspace)
    doc_executor = create_document_tool_executor(workspace)

    vision_initialized = False

    def _ensure_vision_registered() -> None:
        nonlocal vision_initialized
        if not vision_initialized:
            image_loader = ImageLoader(workspace)
            ocr_provider = OllamaOCRProvider(
                base_url=settings.llm_ollama_base_url,
                default_model="qwen2.5vl:3b",
                request_timeout_seconds=max(settings.llm_ollama_request_timeout_seconds, 900),
            )
            vision_provider = OllamaVisionProvider(
                base_url=settings.llm_ollama_base_url,
                default_model="qwen2.5vl:3b",
                request_timeout_seconds=max(settings.llm_ollama_request_timeout_seconds, 900),
            )
            if isinstance(registry, DefaultToolRegistry):
                register_vision_tools(registry, ocr_provider, vision_provider, image_loader)
            vision_initialized = True

    class CompositeToolExecutor(ToolExecutor):
        def __init__(self, registry: ToolRegistry) -> None:
            self._registry = registry
            self._file_executor = file_executor
            self._doc_executor = doc_executor

        @property
        def registry(self) -> ToolRegistry:
            return self._registry

        async def execute(self, call: ToolCall) -> ToolResult:
            if call.tool_name in (LIST_FILES_TOOL.name, READ_FILE_TOOL.name, WRITE_FILE_TOOL.name):
                return await self._file_executor.execute(call)
            if call.tool_name == CREATE_DOCUMENT_TOOL.name:
                return await self._doc_executor.execute(call)
            if call.tool_name in (OCR_IMAGE_TOOL.name, ANALYZE_IMAGE_TOOL.name):
                _ensure_vision_registered()
                callables = getattr(self._registry, "_callables", {})
                fn = callables.get(call.tool_name)
                if fn:
                    try:
                        start_time = time.monotonic()
                        if inspect.iscoroutinefunction(fn):
                            res = await fn(call.arguments)
                        else:
                            res = await asyncio.to_thread(fn, call.arguments)
                        latency = (time.monotonic() - start_time) * 1000
                        return ToolResult(
                            call_id=call.call_id,
                            tool_name=call.tool_name,
                            output=res,
                            error=False,
                            latency_ms=latency,
                        )
                    except Exception as exc:
                        return self._error_result(call, str(exc))
            return self._error_result(call, f"Unknown tool: {call.tool_name}")

        def _error_result(self, call: ToolCall, message: str) -> ToolResult:
            from core.agent.types import ToolResult
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=True,
                error_message=message[:300],
                latency_ms=0.0,
            )

    return CompositeToolExecutor(registry)


def get_model_router() -> ModelRouter:
    """Get the model router from config/models.yaml or built-in defaults."""
    from pathlib import Path
    config_path = Path("config/models.yaml")
    registry = load_default_registry(config_path if config_path.is_file() else None)
    return ModelRouter(registry=registry)


def get_planner() -> Planner:
    return SimplePlanner()


def get_verifier() -> Verifier:
    return SimpleVerifier()


def get_agent_config() -> AgentConfig:
    return AgentConfig(
        max_iterations=settings.agent_max_iterations,
        max_tool_calls=settings.agent_max_tool_calls,
        max_repetitive_tool_calls=settings.agent_max_repetitive_tool_calls,
        execution_timeout_seconds=max(settings.agent_execution_timeout_seconds, 1800.0),
    )


router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskSubmitRequest(BaseModel):
    task: str = Field(..., description="The task description for the agent to execute.")
    task_id: Optional[str] = Field(None, description="Optional custom task ID; if not provided, a UUID is generated.")
    attachments: Optional[List[str]] = Field(
        None,
        description=(
            "Optional workspace-relative attachment names the task references "
            "(e.g. files the user attached and uploaded). Only safe relative "
            "paths are accepted; absolute paths and traversal are rejected."
        ),
    )


@router.post("", response_model=TaskSummary, status_code=status.HTTP_201_CREATED)
async def submit_task(
    request: TaskSubmitRequest,
    response: Response,
    background_tasks: BackgroundTasks,
) -> TaskSummary:
    """Submit a new task for the agent to execute with deduplication protection.
    
    If an identical task was recently submitted or is currently active, returns the existing
    task with 200 OK to prevent duplicate executions against Ollama.
    """
    task_description = request.task.strip()
    if not task_description:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task description must not be empty.",
        )

    workspace = get_workspace()
    attachments: List[str] = []
    for name in (request.attachments or []):
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            resolved = workspace.resolve(name)
        except WorkspaceError as exc:
            logger.warning("Rejecting task attachment %r: %s", name, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid attachment path: {exc}",
            )
        rel = workspace.relative_to(resolved)
        attachments.append(rel or name.strip())

    # Check for duplicate submission
    signature = f"{task_description}::{tuple(sorted(attachments))}"
    duplicate_task_id = await task_store.check_duplicate(signature, window_seconds=5.0)
    if duplicate_task_id:
        existing_rec = await task_store.get_task_record(duplicate_task_id)
        if existing_rec:
            logger.info("Deduplicated task submission for signature -> existing task %s", duplicate_task_id)
            response.status_code = status.HTTP_200_OK
            return existing_rec.to_summary()

    task_id = request.task_id or str(uuid.uuid4())
    await task_store.record_submission(signature, task_id)

    placeholder_state = AgentState(task_id=task_id, task=task_description)
    placeholder_state.status = AgentStatus.PENDING
    placeholder_result = AgentResult.from_state(placeholder_state)

    rec = await task_store.create_task_record(
        task_id=task_id,
        task_description=task_description,
        attachments=attachments,
        result=placeholder_result,
        created_at=time.time(),
        status="queued",
        phase="queued",
        phase_label="Queued in task runner",
    )

    background_tasks.add_task(task_store.run_or_enqueue, task_id, task_description, attachments)
    return rec.to_summary()


@router.get("", response_model=List[TaskSummary])
async def list_tasks() -> List[TaskSummary]:
    """List all submitted tasks sorted newest first."""
    records = await task_store.list_task_records()
    records.sort(key=lambda r: r.created_at, reverse=True)
    return [r.to_summary() for r in records]


@router.get("/{task_id}", response_model=Task)
async def get_task(task_id: str) -> Task:
    """Get the live status or completed result of a task."""
    rec = await task_store.get_task_record(task_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found.",
        )
    return rec.to_task()


@router.post("/{task_id}/cancel", response_model=TaskSummary)
async def cancel_task_endpoint(task_id: str) -> TaskSummary:
    """Cancel a running or queued task safely."""
    rec = await task_store.get_task_record(task_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found.",
        )
    cancelled = await task_store.cancel_task(task_id)
    if cancelled:
        try:
            audit_logger = get_audit_logger()
            audit_logger.log_policy_decision(
                capability="tasks",
                action="cancel",
                decision="allowed",
                reason=f"Task {task_id} cancelled by user",
                correlation_id=task_id,
            )
        except Exception:
            pass
    updated_rec = await task_store.get_task_record(task_id)
    assert updated_rec is not None
    return updated_rec.to_summary()


async def run_agent_task(
    task_id: str,
    task_description: str,
    attachments: Optional[Sequence[str]] = None,
) -> None:
    """Run the agent task and update task store in real-time."""
    try:
        # Phase 2: Deterministic Fast-Path Classification (bypasses LLM for greetings/thanks/pings)
        intent, fast_response = classify_intent(task_description, attachments)
        if intent == TaskIntent.FAST and fast_response is not None:
            try:
                audit_logger = get_audit_logger()
                audit_logger.log_policy_decision(
                    capability="tasks",
                    action="fast_path_response",
                    decision="allowed",
                    reason="Task classified as deterministic FAST path (no LLM required)",
                    correlation_id=task_id,
                )
            except Exception:
                pass

            state = AgentState(task_id=task_id, task=task_description)
            state.status = AgentStatus.COMPLETE
            state.current_phase = ExecutionPhase.COMPLETE
            state.messages.append(AgentMessage(role="user", content=task_description))
            state.messages.append(AgentMessage(role="assistant", content=fast_response))
            result = AgentResult.from_state(state)
            completed_at = time.time()

            await task_store.update_task_state(
                task_id,
                status="completed",
                phase="completed",
                phase_label="Completed",
                model="fast-path",
                streaming_content=fast_response,
                result=result,
                completed_at=completed_at,
            )
            return

        workspace = get_workspace()
        registry = get_tool_registry()
        executor = get_tool_executor(registry, workspace)
        router = get_model_router()
        gateway = get_model_service().gateway
        planner = get_planner()
        verifier = get_verifier()
        config = get_agent_config()

        agent = Agent(
            model_router=router,
            model_gateway=gateway,
            tool_executor=executor,
            planner=planner,
            verifier=verifier,
            config=config,
            tool_registry=registry,
        )

        def on_agent_event(event: dict[str, Any]) -> None:
            event_type = event.get("type")
            if event_type == "phase":
                raw_phase = event.get("phase", "")
                phase_map = {
                    "understand": ("understanding", "🧠 Understanding request..."),
                    "plan": ("planning", "📋 Planning steps..."),
                    "route_model": ("routing", "🔀 Selecting local model..."),
                    "generate_response": ("generating", "📝 Generating response..."),
                    "decide_action": ("deciding", "🔎 Analyzing next step..."),
                    "tool_request": ("tool_request", "🛡️ Checking security policy..."),
                    "tool_execution": ("tool_execution", "🔧 Executing tool..."),
                    "observation": ("observing", "📊 Processing tool observation..."),
                    "verify": ("verifying", "✅ Verifying response..."),
                    "complete": ("completed", "✅ Completed"),
                    "failed": ("failed", "❌ Task failed"),
                }
                mapped_phase, mapped_label = phase_map.get(
                    raw_phase, (raw_phase, f"Processing {raw_phase}...")
                )
                task_store.update_streaming_state(
                    task_id, phase=mapped_phase, phase_label=mapped_label
                )
            elif event_type == "model_selected":
                model_name = event.get("model")
                task_store.update_streaming_state(
                    task_id,
                    model=model_name,
                    phase="routing",
                    phase_label=f"🔀 Local model selected: {model_name}",
                )
            elif event_type == "thinking_token":
                accumulated = event.get("accumulated", "")
                task_store.update_streaming_state(
                    task_id,
                    streaming_thinking=accumulated,
                    phase="generating",
                    phase_label="🧠 Thinking...",
                )
            elif event_type == "token":
                accumulated = event.get("accumulated", "")
                task_store.update_streaming_state(
                    task_id,
                    streaming_content=accumulated,
                    phase="generating",
                    phase_label="📝 Generating response...",
                )
            elif event_type == "generating_start":
                model_name = event.get("model") or "local model"
                task_store.update_streaming_state(
                    task_id,
                    phase="generating",
                    phase_label=f"📝 Generating response with {model_name}...",
                )
            elif event_type == "tool_start":
                tool_name = event.get("tool", "tool")
                call_id = event.get("call_id", f"{task_id}-tool")
                tool_event = ToolEvent(
                    id=call_id,
                    label=tool_name,
                    status="running",
                    detail="Executing...",
                )
                task_store.update_streaming_state(
                    task_id,
                    phase="tool_execution",
                    phase_label=f"🔧 Running {tool_name}...",
                )
                task_store.update_tool_event(task_id, tool_event)
            elif event_type == "tool_end":
                tool_name = event.get("tool", "tool")
                call_id = event.get("call_id", f"{task_id}-tool")
                is_err = event.get("error", False)
                output = event.get("output")
                tool_event = ToolEvent(
                    id=call_id,
                    label=tool_name,
                    status="failed" if is_err else "succeeded",
                    detail=output or ("Failed" if is_err else "Completed"),
                )
                task_store.update_tool_event(task_id, tool_event)
            elif event_type == "verify_start":
                task_store.update_streaming_state(
                    task_id,
                    phase="verifying",
                    phase_label="🛡️ Verifying output...",
                )

        result = await agent.run(
            task_description,
            task_id=task_id,
            attachments=attachments,
            on_event=on_agent_event,
        )

        completed_at = time.time()
        has_assistant_content = any(
            getattr(m, "role", None) in ("assistant", "agent")
            and bool(getattr(m, "content", None) and str(getattr(m, "content", "")).strip())
            for m in (result.messages or [])
        )
        if result.status == AgentStatus.COMPLETE and has_assistant_content and not result.error:
            final_status = "completed"
            phase_label = "Completed"
        else:
            final_status = "failed"
            if not has_assistant_content:
                phase_label = "Generation interrupted: No final answer produced"
            else:
                phase_label = result.error or "Task failed"

        await task_store.update_task_state(
            task_id,
            status=final_status,
            phase="completed" if final_status == "completed" else "failed",
            phase_label=phase_label,
            result=result,
            completed_at=completed_at,
        )

    except asyncio.CancelledError:
        logger.info("Agent task %s was cancelled", task_id)
        state = AgentState(task_id=task_id, task=task_description)
        state.status = AgentStatus.CANCELLED
        result = AgentResult.from_state(state, error="Task cancelled by user")
        completed_at = time.time()
        await task_store.update_task_state(
            task_id,
            status="cancelled",
            phase="cancelled",
            phase_label="Cancelled by user",
            result=result,
            completed_at=completed_at,
        )
        raise
    except Exception as exc:
        logger.exception("Failed to run agent task %s: %s", task_id, exc)
        state = AgentState(task_id=task_id, task=task_description)
        state.status = AgentStatus.FAILED
        state.errors.append(f"{type(exc).__name__}: {str(exc)}")
        result = AgentResult.from_state(state, error=f"{type(exc).__name__}: {exc}")
        completed_at = time.time()
        await task_store.update_task_state(
            task_id,
            status="failed",
            phase="failed",
            phase_label=f"Error: {type(exc).__name__}",
            result=result,
            completed_at=completed_at,
            error=str(exc),
        )