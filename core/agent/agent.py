"""The Agent Runtime — a controlled, stateful, tool-using agent.

The agent implements a deterministic state machine that:

1. **START** → UNDERSTAND
2. **UNDERSTAND** → validates the task
3. **UNDERSTAND** → PLAN
4. **PLAN** → calls the planner
5. **PLAN** → ROUTE_MODEL
6. **ROUTE_MODEL** → selects a model via the ModelRouter
7. **ROUTE_MODEL** → DECIDE_ACTION
8. **DECIDE_ACTION** → selects the next step from the plan (or generates a model response)
9. **DECIDE_ACTION** → TOOL_REQUEST (if a tool step is next) or COMPLETE (if plan is empty)
10. **TOOL_REQUEST** → records the tool call
11. **TOOL_REQUEST** → TOOL_EXECUTION
12. **TOOL_EXECUTION** → runs the tool via the ToolExecutor
13. **TOOL_EXECUTION** → OBSERVATION
14. **OBSERVATION** → records the result
15. **OBSERVATION** → CONTINUE? (if steps remain) → DECIDE_ACTION or → VERIFY
16. **VERIFY** → calls the verifier
17. **VERIFY** → COMPLETE or → DECIDE_ACTION (retry)

The agent enforces explicit safety limits: max_iterations, max_tool_calls,
max_repetitive_tool_calls, and execution_timeout_seconds. It never allows an
uncontrolled infinite loop.
"""
from __future__ import annotations

import asyncio
import inspect
import re
import logging
import time
import uuid
from typing import Any, Callable, Optional, Sequence

from core.agent.errors import (
    AgentExecutionError,
    AgentTimeoutError,
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
from core.agent.interfaces import Planner, ToolExecutor, ToolRegistry, Verifier
from core.agent.types import (
    AgentConfig,
    AgentMessage,
    AgentResult,
    AgentState,
    AgentStatus,
    ExecutionPhase,
    Observation,
    Plan,
    PlanStep,
    ToolCall,
    ToolResult,
)
from core.llm.gateway import ModelGateway
from core.llm.types import ChatMessage, GenerationRequest
from core.routing import (
    ModelRouter,
    NoSuitableModelError,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    TaskType,
)
from core.security.policy_engine import get_policy_engine
from core.audit_logger import get_audit_logger


_logger = logging.getLogger("sovereign-ai.agent")


class Agent:
    """A controlled, stateful, tool-using agent.

    The agent is constructed with its collaborators (router, planner, executor,
    verifier) and a :class:`AgentConfig`. It is safe for concurrent use at
    the instance level only — each call to :meth:`run` gets its own
    :class:`AgentState`.

    The agent never talks to Ollama or any provider directly. It uses a
    :class:`core.llm.ModelGateway` for generation. It uses a
    :class:`core.routing.ModelRouter` for model selection.

    Example::

        agent = Agent(
            model_router=router,
            model_gateway=gateway,
            tool_executor=executor,
            planner=planner,
            verifier=verifier,
            config=AgentConfig(max_iterations=10),
        )
        result = await agent.run("Explain how photosynthesis works.")
        print(result.status, result.selected_model)
    """

    def __init__(
        self,
        model_router: ModelRouter,
        model_gateway: ModelGateway,
        tool_executor: ToolExecutor,
        planner: Planner,
        verifier: Verifier,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        on_event: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        if model_router is None:
            raise TypeError("Agent requires a ModelRouter")
        if model_gateway is None:
            raise TypeError("Agent requires a ModelGateway")
        if tool_executor is None:
            raise TypeError("Agent requires a ToolExecutor")
        if planner is None:
            raise TypeError("Agent requires a Planner")
        if verifier is None:
            raise TypeError("Agent requires a Verifier")

        self._router = model_router
        self._gateway = model_gateway
        self._executor = tool_executor
        self._planner = planner
        self._verifier = verifier
        self._config = config or AgentConfig()
        self._on_event = on_event
        self._active_on_event: Optional[Callable[[dict[str, Any]], Any]] = None
        # The tool registry is optional: if not given, the agent falls back
        # to reading tool names from the executor (if it has a ``registry``
        # attribute, as SyncToolExecutor does).
        self._tool_registry = tool_registry

    def _tool_names(self) -> list[str]:
        """Return the list of registered tool names.

        Prefers an explicitly-passed registry. Falls back to the executor's
        attached registry (when using a SyncToolExecutor). If neither is
        available, returns an empty list.
        """
        if self._tool_registry is not None:
            return self._tool_registry.names()
        registry = getattr(self._executor, "registry", None)
        if registry is not None:
            return registry.names()
        return []

    def _emit_event(self, event_type: str, data: Optional[dict[str, Any]] = None) -> None:
        """Emit a structured lifecycle event if a callback is registered."""
        callback = self._active_on_event or self._on_event
        if callback is None:
            return
        payload = {"type": event_type, "timestamp": time.time(), **(data or {})}
        try:
            res = callback(payload)
            if inspect.isawaitable(res):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(res)
                except RuntimeError:
                    pass
        except Exception as exc:
            _logger.warning("agent.on_event_error: %s", exc)

    # ------------------------------------------------------------------ Run

    async def run(
        self,
        task: str,
        task_id: Optional[str] = None,
        attachments: Optional[Sequence[str]] = None,
        on_event: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        """Run the agent on the given task.

        This is the main entry point. The agent moves through its state machine
        and returns an :class:`AgentResult` when done.

        Args:
            task: The user's task string. Must be non-empty.
            task_id: Optional unique identifier for this run. A UUID is generated
                if not provided.
            attachments: Optional workspace-relative attachment names the task
                references (e.g. uploaded files). They are stored on the agent
                state so the planner can produce correct tool call arguments
                (for example the ``path`` for ``read_file``). These must already
                be validated workspace-relative paths by the caller.

        Returns:
            An :class:`AgentResult` snapshot of the final state.

        Raises:
            InvalidTaskError: if the task is empty or whitespace-only.
            AgentTimeoutError: if the execution exceeds the configured deadline.
            RoutingFailureError: if the model router cannot select a model.
            ToolExecutionError: if a tool fails and retries are disabled.
            RepetitiveToolCallError: if the same tool is called too many times.
            MaxToolCallsExceededError: if the tool call limit is reached.
            IterationLimitExceededError: if the iteration limit is reached.
            VerificationFailureError: if the verifier rejects the output.
        """
        task = task.strip()
        if not task:
            raise InvalidTaskError("Task must be a non-empty string")

        task_id = task_id or str(uuid.uuid4())
        self._active_on_event = on_event or self._on_event
        state = AgentState(
            task_id=task_id,
            task=task,
            attachments=list(attachments or ()),
        )
        state.status = AgentStatus.RUNNING
        state.started_at = time.monotonic()

        _logger.info(
            "agent.start  task_id=%s  task_length=%d  max_iterations=%d  max_tool_calls=%d",
            task_id,
            len(task),
            self._config.max_iterations,
            self._config.max_tool_calls,
        )
        self._emit_event("start", {"task_id": task_id, "task": task})

        try:
            state = await self._execute(state)
            state.status = AgentStatus.COMPLETE
            state.current_phase = ExecutionPhase.COMPLETE
            _logger.info(
                "agent.complete  task_id=%s  iterations=%d  tool_calls=%d  status=complete",
                task_id,
                state.iteration,
                len(state.tool_calls),
            )
            self._emit_event("complete", {"task_id": task_id, "iterations": state.iteration})
            return AgentResult.from_state(state)
        except AgentTimeoutError:
            state.status = AgentStatus.TIMEOUT
            state.current_phase = ExecutionPhase.FAILED
            _logger.warning("agent.timeout  task_id=%s  iterations=%d", task_id, state.iteration)
            self._emit_event("failed", {"task_id": task_id, "error": "AgentTimeoutError"})
            return AgentResult.from_state(state, error="AgentTimeoutError")
        except AgentExecutionError as exc:
            state.status = AgentStatus.FAILED
            state.current_phase = ExecutionPhase.FAILED
            state.errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            _logger.warning(
                "agent.fail  task_id=%s  error_type=%s  iterations=%d",
                task_id,
                type(exc).__name__,
                state.iteration,
            )
            self._emit_event("failed", {"task_id": task_id, "error": type(exc).__name__})
            return AgentResult.from_state(state, error=type(exc).__name__)
        finally:
            self._active_on_event = None

    # ------------------------------------------------------------------ State machine

    async def _execute(self, state: AgentState) -> AgentState:
        """Run the state machine until a terminal state is reached."""
        deadline = state.started_at + self._config.execution_timeout_seconds

        # --- Phase: START → UNDERSTAND → PLAN ---
        state = self._transition(state, ExecutionPhase.UNDERSTAND)
        state.messages.append(AgentMessage(role="user", content=state.task))
        state = self._transition(state, ExecutionPhase.PLAN)

        # --- Phase: PLANNING ---
        try:
            plan = await self._planner.plan(
                task=state.task,
                available_tools=self._tool_names(),
                state=state,
            )
        except Exception as exc:
            raise PlanningFailureError(
                f"Planner failed: {exc}",
                cause=exc,
            ) from exc

        state.plan = plan
        if not plan.is_empty:
            _logger.debug(
                "agent.plan  task_id=%s  steps=%d  first_step=%s",
                state.task_id,
                len(plan.steps),
                plan.steps[0].tool_name if plan.steps else None,
            )

        # --- Phase: ROUTE_MODEL ---
        state = self._transition(state, ExecutionPhase.ROUTE_MODEL)
        state.selected_model = await self._route(state)
        _logger.info(
            "agent.model_selected  task_id=%s  model=%s",
            state.task_id,
            state.selected_model,
        )

        # --- Direct-answer path: if plan is empty, generate a response ---
        if state.plan is not None and state.plan.is_empty:
            # --- Phase: GENERATE_RESPONSE ---
            state = self._transition(state, ExecutionPhase.GENERATE_RESPONSE)
            self._emit_event("generating_start", {"model": state.selected_model, "task_id": state.task_id})

            task_lower = state.task.lower()
            is_multi_question = any(
                f"{i}." in task_lower for i in range(1, 10)
            ) or any(
                w in task_lower for w in (
                    "manual", "guide", "detailed", "20-paragraph", "comprehensive",
                    "in-depth", "thorough", "tutorial", "essay", "report", "article",
                    "solve all", "5 questions", "five questions", "all questions", "basics of c++",
                    "introduction", "comparison table", "advantages", "disadvantages",
                )
            )
            is_simple_greeting = task_lower in ("hi", "hello", "hey", "hi there", "hello there", "test")
            is_simple_question = len(state.task) <= 150 and not is_multi_question and not is_simple_greeting

            if is_multi_question:
                # Complex/multi-part request: large token budget
                max_tokens = 4096
                num_ctx = 16384
                enable_think = True
                system_instruction = ChatMessage.system(
                    "You are SovereignAI, an expert technical and educational AI assistant. "
                    "Think briefly, then provide a thorough, comprehensive, and complete answer "
                    "fulfilling all user instructions. Answer every numbered question completely without omitting any parts."
                )
            elif is_simple_greeting:
                max_tokens = 1024
                num_ctx = 2048
                enable_think = False
                system_instruction = ChatMessage.system(
                    "You are SovereignAI, a helpful AI assistant. Greet the user in 1 short, friendly sentence."
                )
            elif is_simple_question:
                # Short questions: need adequate tokens for thinking + complete answer
                max_tokens = 4096
                num_ctx = 8192
                enable_think = False
                system_instruction = ChatMessage.system(
                    "You are SovereignAI, an expert technical AI assistant. "
                    "Provide a direct, clear, and complete answer to the user's request."
                )
            else:
                # Normal explanations, code questions, etc.
                max_tokens = 4096
                num_ctx = 8192
                enable_think = False
                system_instruction = ChatMessage.system(
                    "You are SovereignAI, an expert technical AI assistant. "
                    "Provide a clear, direct, and complete answer. Be thorough but concise."
                )

            generation_request = GenerationRequest(
                messages=[system_instruction, ChatMessage.user(state.task)],
                model=state.selected_model,
                max_tokens=max_tokens,
                temperature=0.3,
                num_ctx=num_ctx,
                keep_alive="60m",
                think=enable_think,
            )
            thinking_chunks: list[str] = []
            content_chunks: list[str] = []
            try:
                async for chunk in self._gateway.stream(generation_request):
                    if getattr(chunk, "is_thinking", False):
                        thinking_chunks.append(str(chunk))
                        accum_thinking = "".join(thinking_chunks)
                        self._emit_event(
                            "thinking_token",
                            {
                                "token": str(chunk),
                                "accumulated": accum_thinking,
                                "task_id": state.task_id,
                            },
                        )
                    else:
                        content_chunks.append(str(chunk))
                        accum = "".join(content_chunks)
                        if "</think>" in accum:
                            clean_accum = re.sub(r"^.*?<\/think>", "", accum, flags=re.DOTALL).lstrip()
                            if clean_accum:
                                self._emit_event(
                                    "token",
                                    {
                                        "token": str(chunk),
                                        "accumulated": clean_accum,
                                        "task_id": state.task_id,
                                    },
                                )
                        elif "<think>" not in accum and not accum.startswith("<think>"):
                            self._emit_event(
                                "token",
                                {
                                    "token": str(chunk),
                                    "accumulated": accum,
                                    "task_id": state.task_id,
                                },
                            )
            except Exception as stream_err:
                _logger.warning("agent.stream_fallback  task_id=%s  err=%s", state.task_id, stream_err)
                if not content_chunks:
                    generation_response = await self._gateway.generate(generation_request)
                    content_chunks.append(generation_response.content)
                    if generation_response.raw and isinstance(generation_response.raw, dict):
                        resp_msg = generation_response.raw.get("message", {})
                        if isinstance(resp_msg, dict) and resp_msg.get("thinking"):
                            thinking_chunks.append(resp_msg["thinking"])

            raw_content = "".join(content_chunks)
            self._emit_event("generating_done", {"model": state.selected_model, "task_id": state.task_id})
            thinking_text = "".join(thinking_chunks).strip() if thinking_chunks else None
            if not thinking_text and "<think>" in raw_content:
                think_match = re.search(r"<think>(.*?)(?:<\/think>|$)", raw_content, flags=re.DOTALL)
                if think_match:
                    thinking_text = think_match.group(1).strip()
            
            content = raw_content
            if "</think>" in content:
                content = re.sub(r"^.*?<\/think>", "", content, flags=re.DOTALL).strip()
            elif "<think>" in content:
                if "</think>" in content:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                else:
                    # Model stopped before closing </think>
                    if not thinking_text:
                        thinking_text = content.removeprefix("<think>").strip()
                    content = ""

            # If the model produced internal thinking but ran out of tokens before writing the final answer:
            if not content.strip() and thinking_text:
                _logger.info("agent.recover_thinking_answer  task_id=%s", state.task_id)
                recovery_req = GenerationRequest(
                    messages=[
                        ChatMessage.system(
                            "You are SovereignAI. Based on the reasoning provided in the thinking block, "
                            "provide the direct, complete final answer to the user. Do not output any <think> tags."
                        ),
                        ChatMessage.user(state.task),
                        ChatMessage.assistant(f"<think>\n{thinking_text}\n</think>\n"),
                    ],
                    model=state.selected_model,
                    max_tokens=2048,
                    temperature=0.3,
                    num_ctx=8192,
                    keep_alive="60m",
                    think=False,
                )
                try:
                    recovery_resp = await self._gateway.generate(recovery_req)
                    recovered_content = recovery_resp.content.strip()
                    if "</think>" in recovered_content:
                        recovered_content = re.sub(r"^.*?<\/think>", "", recovered_content, flags=re.DOTALL).strip()
                    if recovered_content:
                        content = recovered_content
                        self._emit_event(
                            "token",
                            {
                                "token": content,
                                "accumulated": content,
                                "task_id": state.task_id,
                            },
                        )
                except Exception as rec_err:
                    _logger.warning("agent.recovery_failed  err=%s", rec_err)

            # Secondary fallback: if recovery failed, extract substantive thoughts as response
            if not content.strip() and thinking_text:
                # Find conclusion or final paragraph in thinking
                paragraphs = [p.strip() for p in thinking_text.split("\n\n") if p.strip()]
                if paragraphs:
                    # Use the last paragraph(s) as the synthesized response
                    content = paragraphs[-1]
                    self._emit_event(
                        "token",
                        {
                            "token": content,
                            "accumulated": content,
                            "task_id": state.task_id,
                        },
                    )

            # Append the generated response to the agent's messages
            state.messages.append(
                AgentMessage(role="assistant", content=content, thinking=thinking_text)
            )
            # Move to verification
            state = self._transition(state, ExecutionPhase.VERIFY)
            state = await self._verify(state)
            if state.verification_result and state.verification_result.passed and content.strip():
                # Move to complete
                state.status = AgentStatus.COMPLETE
                state = self._transition(state, ExecutionPhase.COMPLETE)
                return state
            else:
                # Verification failed or content is empty / thinking-only
                state.status = AgentStatus.FAILED
                err_msg = (
                    state.verification_result.reason
                    if state.verification_result
                    else "Generation produced empty response."
                )
                state.errors.append(err_msg)
                state = self._transition(state, ExecutionPhase.FAILED)
                return state

        # --- Main loop: DECIDE_ACTION → TOOL_EXECUTION → OBSERVATION → loop ---
        while True:
            # --- Deadline check ---
            if time.monotonic() > deadline:
                raise AgentTimeoutError()

            # --- Iteration check ---
            state.iteration += 1
            if state.iteration > self._config.max_iterations:
                raise IterationLimitExceededError(self._config.max_iterations)

            # --- CONTINUE? (DECIDE_ACTION) ---
            state = self._transition(state, ExecutionPhase.DECIDE_ACTION)

            # If the plan is empty or all steps are done, move to VERIFY.
            if state.plan is None or state.plan.is_empty or state.current_step >= len(state.plan.steps):
                state = self._transition(state, ExecutionPhase.VERIFY)
                state = await self._verify(state)
                if state.verification_result and state.verification_result.passed:
                    break  # → COMPLETE
                else:
                    # Verification failed. Decide whether to re-plan (which
                    # counts as a new iteration that will be checked at the
                    # top of the loop) or to surface the failure.
                    # If the next loop iteration would exceed the cap, raise
                    # the iteration limit error — this is the more specific
                    # and informative failure mode.
                    if state.iteration + 1 > self._config.max_iterations:
                        raise IterationLimitExceededError(self._config.max_iterations)
                    # Re-plan and try once more.
                    state = self._transition(state, ExecutionPhase.PLAN)
                    try:
                        new_plan = await self._planner.plan(
                            task=state.task,
                            available_tools=self._tool_names(),
                            state=state,
                        )
                        state.plan = new_plan
                        state.current_step = 0
                    except Exception:
                        raise VerificationFailureError(
                            "Verifier rejected the agent's final output "
                            "and the planner could not produce a new plan."
                        )
                    continue  # restart DECIDE_ACTION loop

            # --- NEXT STEP: TOOL_REQUEST ---
            step = state.plan.steps[state.current_step]  # type: ignore[index]
            state = self._transition(state, ExecutionPhase.TOOL_REQUEST)
            tool_call = self._make_tool_call(state, step)
            state.tool_calls.append(tool_call)

            # --- Tool call limit ---
            if len(state.tool_calls) >= self._config.max_tool_calls:
                raise MaxToolCallsExceededError(self._config.max_tool_calls)

            # --- Repetition check ---
            self._check_repetitive(state, tool_call)

            _logger.info(
                "agent.tool_request  task_id=%s  iteration=%d  tool=%s  call_id=%s",
                state.task_id,
                state.iteration,
                tool_call.tool_name,
                tool_call.call_id,
            )

            # --- POLICY EVALUATION (before any tool/provider/sandbox run) ---
            if self._policy_blocked(state, tool_call):
                continue  # → DECIDE_ACTION

            # Log tool execution start
            try:
                audit_logger = get_audit_logger()
                audit_logger.log_tool_execution_start(
                    tool_name=tool_call.tool_name,
                    correlation_id=tool_call.call_id,
                )
            except Exception:
                # Ignore audit logging errors to prevent breaking the agent
                pass

            # --- TOOL_EXECUTION ---
            state = self._transition(state, ExecutionPhase.TOOL_EXECUTION)
            self._emit_event("tool_start", {"tool": tool_call.tool_name, "call_id": tool_call.call_id, "task_id": state.task_id})
            try:
                tool_result = await self._executor.execute(tool_call)
            except UnknownToolError as exc:
                self._emit_event("tool_end", {"tool": exc.tool_name, "call_id": tool_call.call_id, "error": True, "task_id": state.task_id})
                # Planner asked for a non-existent tool — this is a planning error.
                state.errors.append(f"Unknown tool: {exc.tool_name}")
                raise ToolExecutionError(
                    tool_name=exc.tool_name,
                    message="Tool referenced by plan is not registered",
                    cause=exc,
                ) from exc

            self._emit_event("tool_end", {
                "tool": tool_result.tool_name,
                "call_id": tool_result.call_id,
                "error": bool(tool_result.error),
                "output": str(tool_result.output)[:200] if tool_result.output is not None else None,
                "task_id": state.task_id,
            })

            # Log tool execution end
            try:
                audit_logger = get_audit_logger()
                audit_logger.log_tool_execution_end(
                    tool_name=tool_call.tool_name,
                    correlation_id=tool_call.call_id,
                    success=not tool_result.error,
                    latency_ms=tool_result.latency_ms,
                    error=tool_result.error_message if tool_result.error else None,
                )
            except Exception:
                # Ignore audit logging errors to prevent breaking the agent
                pass

            if tool_result.error:
                _logger.warning(
                    "agent.tool_failed  task_id=%s  tool=%s  call_id=%s  error=%s",
                    state.task_id,
                    tool_result.tool_name,
                    tool_result.call_id,
                    tool_result.error_message,
                )
                # Record the failure observation and continue.
                # The agent can decide to retry or skip based on the plan.
                state.observations.append(
                    Observation(
                        call_id=tool_result.call_id,
                        content=f"[Tool error] {tool_result.error_message}",
                        source=tool_result.tool_name,
                    )
                )
                state.current_step += 1
                state.completed_steps += 1
                continue  # → DECIDE_ACTION

            _logger.info(
                "agent.tool_done  task_id=%s  tool=%s  call_id=%s  latency_ms=%.1f",
                state.task_id,
                tool_result.tool_name,
                tool_result.call_id,
                tool_result.latency_ms,
            )

            # --- OBSERVATION ---
            state = self._transition(state, ExecutionPhase.OBSERVATION)
            obs = Observation(
                call_id=tool_result.call_id,
                content=str(tool_result.output) if tool_result.output is not None else "",
                source=tool_result.tool_name,
            )
            state.observations.append(obs)

            # Advance to next step.
            state.current_step += 1
            state.completed_steps += 1

        # The plan is exhausted and verification passed. A run that ended with
        # an unrecovered tool failure was NOT genuinely completed — surface it
        # as a failed task instead of a false "completed". Policy denials and
        # approval requests are not tool failures, and a failure is considered
        # recovered when a later attempt on the same tool succeeded.
        if self._has_unrecovered_tool_failure(state):
            failed = self._failed_tool_names(state)
            _logger.warning(
                "agent.unrecovered_tool_failure  task_id=%s  tools=%s",
                state.task_id,
                failed,
            )
            raise ToolExecutionError(
                tool_name=", ".join(failed) or "tool",
                message=(
                    "A tool call failed and was not recovered before the task "
                    "finished; the task cannot be reported as completed."
                ),
            )

        # If the agent used tools, synthesize a final answer from the collected
        # observations so the user receives a genuine response (e.g. a file
        # summary), not just raw tool output.
        if state.observations and not any(
            m.role == "assistant" for m in state.messages
        ):
            try:
                state = await self._synthesize_final_response(state)
            except Exception:  # pragma: no cover - defensive
                # Never let finalization failure crash a run whose tools already
                # succeeded; the observations remain the source of truth.
                _logger.warning(
                    "agent.final_response_failed  task_id=%s",
                    state.task_id,
                )

        # Check completion invariant before marking COMPLETE:
        # A task is only complete if an assistant response with valid content exists
        has_assistant_content = any(
            getattr(m, "role", None) in ("assistant", "agent")
            and bool(getattr(m, "content", None) and str(getattr(m, "content", "")).strip())
            for m in state.messages
        )
        if not has_assistant_content:
            state.status = AgentStatus.FAILED
            state.errors.append("Task finished without generating a final assistant response.")
            state = self._transition(state, ExecutionPhase.FAILED)
            return state

        # --- COMPLETE ---
        state.status = AgentStatus.COMPLETE
        state = self._transition(state, ExecutionPhase.COMPLETE)
        return state

    # ------------------------------------------------------------------ Helpers

    def _transition(self, state: AgentState, phase: ExecutionPhase) -> AgentState:
        """Move to a new phase."""
        state.current_phase = phase
        _logger.debug(
            "agent.phase  task_id=%s  from=%s  to=%s",
            state.task_id,
            phase.value,
            phase.value,
        )
        self._emit_event("phase", {"phase": phase.value, "task_id": state.task_id})
        return state

    def _detect_task_type(self, state: AgentState) -> TaskType:
        """Derive the appropriate TaskType from the user task, plan, and attachments."""
        task_lower = state.task.lower()

        # 1. Vision: image attachments or explicit vision/OCR keywords
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff")
        if any(a.lower().endswith(image_exts) for a in state.attachments):
            return TaskType.VISION
        if any(ext in task_lower for ext in image_exts):
            return TaskType.VISION
        if any(kw in task_lower for kw in ("ocr", "read image", "transcribe image", "describe picture")):
            return TaskType.VISION

        # 2. Coding: tool execute_code in plan, or code-centric programming keywords
        if state.plan and any(s.tool_name == "execute_code" for s in state.plan.steps):
            return TaskType.CODING
        code_keywords = (
            "write code", "implement a function", "write a function", "write a c++",
            "write a c++ program", "c++ program", "factorial", "write a program",
            "write a cpp", "cpp program", "write python", "write script",
            "c++ function", "python function", "reverse a string", "fibonacci",
            "algorithm", "debug code", "refactor", "unit test", "write a class",
            "implement class", "reverse string", "sorting algorithm", "binary search"
        )
        if any(kw in task_lower for kw in code_keywords):
            return TaskType.CODING

        # 3. Document Analysis: document/file attachments, read_file tool, or summarization
        if state.attachments or (state.plan and any(s.tool_name == "read_file" for s in state.plan.steps)):
            return TaskType.DOCUMENT
        doc_keywords = (
            "summarize", "summary", "document", ".txt", ".md", ".docx", ".pdf",
            "bullet points", "takeaways", "read file", "read the file"
        )
        if any(kw in task_lower for kw in doc_keywords):
            return TaskType.DOCUMENT

        # 4. Reasoning: analytical comparison and multi-step deduction
        reasoning_keywords = ("compare and contrast", "pros and cons", "evaluate the differences", "logical deduction")
        if any(kw in task_lower for kw in reasoning_keywords):
            return TaskType.REASONING

        # 5. General Chat / explanation
        return TaskType.CHAT

    async def _route(self, state: AgentState) -> str:
        """Route to select a model for the current task.

        The agent derives a :class:`RoutingRequest` from the task and plan,
        asks the ModelRouter, and returns the selected **provider** model name
        (``decision.model.provider_model``) — the model identifier the gateway
        passes to the provider. The router's logical-name abstraction remains
        available on ``decision.model.logical_name``.
        """
        task_type = self._detect_task_type(state)
        request = RoutingRequest(
            task_type=task_type,
            required_capabilities=frozenset(),
            input_modalities=frozenset(),
        )
        try:
            route_result = self._router.route(request)
        except NoSuitableModelError as exc:
            if task_type != TaskType.CHAT:
                # Fallback to TaskType.CHAT if specialized model is not configured
                request = RoutingRequest(
                    task_type=TaskType.CHAT,
                    required_capabilities=frozenset(),
                    input_modalities=frozenset(),
                )
                try:
                    route_result = self._router.route(request)
                except NoSuitableModelError:
                    raise RoutingFailureError(
                        f"ModelRouter could not select a model: {exc.reason}",
                        cause=exc,
                    ) from exc
            else:
                raise RoutingFailureError(
                    f"ModelRouter could not select a model: {exc.reason}",
                    cause=exc,
                ) from exc
        except RoutingError as exc:
            raise RoutingFailureError(
                f"ModelRouter error: {exc}",
                cause=exc,
            ) from exc

        if inspect.isawaitable(route_result):
            decision = await route_result
        else:
            decision = route_result
        selected = decision.model.provider_model
        self._emit_event("model_selected", {"model": selected, "task_id": state.task_id})
        return selected

    def _has_unrecovered_tool_failure(self, state: AgentState) -> bool:
        """Return True if the run ended with a tool failure that was never recovered.

        A tool name is considered *recovered* when its most recent executed
        outcome was a success. Policy denials / approval requests are recorded
        as observations but are not tool failures.
        """
        obs_by_call_id = {obs.call_id: str(obs.content) for obs in state.observations}
        last_ok: dict[str, bool] = {}
        for call in state.tool_calls:
            content = obs_by_call_id.get(call.call_id)
            if content is None:
                continue
            if content.startswith("[Tool error]"):
                last_ok[call.tool_name] = False
            elif content.startswith(("[Policy Denied]", "[Approval Required]")):
                continue  # permission state, not a tool failure
            else:
                last_ok[call.tool_name] = True
        return any(not ok for ok in last_ok.values())

    def _failed_tool_names(self, state: AgentState) -> list[str]:
        """Return the sorted tool names that ended in an unrecovered failure."""
        obs_by_call_id = {obs.call_id: str(obs.content) for obs in state.observations}
        failed: set[str] = set()
        for call in state.tool_calls:
            content = obs_by_call_id.get(call.call_id)
            if content is not None and content.startswith("[Tool error]"):
                failed.add(call.tool_name)
        return sorted(failed)

    async def _synthesize_final_response(self, state: AgentState) -> AgentState:
        """Generate a closing assistant message from the task and observations.

        Used after tool execution so the agent answers the user's request (for
        example summarising an attached file or solving questions from an image)
        instead of returning only raw tool output.
        """
        state = self._transition(state, ExecutionPhase.GENERATE_RESPONSE)
        self._emit_event("generating_start", {"model": state.selected_model, "task_id": state.task_id})
        messages = [ChatMessage.user(state.task)]
        context = "\n".join(
            f"[{obs.source}] {obs.content}" for obs in state.observations
        )
        if context:
            # Preserve full tool results unless extremely large (>12000 chars)
            if len(context) > 12000:
                context = context[:8000] + "\n...[truncated for brevity]...\n" + context[-3000:]
            messages.append(ChatMessage.user(f"Tool results / Extracted content:\n{context}"))

        task_lower = state.task.lower()
        is_multi_question = any(
            f"{i}." in task_lower for i in range(1, 10)
        ) or any(
            w in task_lower for w in (
                "solve all", "all questions", "questions", "problems", "exercises",
                "c++", "python", "programming", "manual", "guide", "detailed",
                "comprehensive", "thorough", "in-depth",
            )
        )
        # Synthesis after tool use: raise budgets and disable thinking.
        # The model already has tool context — it should write the answer, not reason.
        max_tokens = 4096 if is_multi_question else 2048

        if is_multi_question:
            system_instruction = ChatMessage.system(
                "You are SovereignAI, an expert technical AI assistant. "
                "Analyze the extracted tool results and answer ALL requested questions completely and thoroughly. "
                "Do not skip or stop before any question is answered. Maintain the question numbers."
            )
        else:
            system_instruction = ChatMessage.system(
                "You are a helpful, concise AI assistant. Provide a direct, factual summary based on the tool results. "
                "Be clear and to the point."
            )

        request = GenerationRequest(
            messages=[system_instruction] + messages,
            model=state.selected_model,
            max_tokens=max_tokens,
            temperature=0.2,
            num_ctx=16384 if len(context) > 4000 else 8192,
            keep_alive="60m",
            think=False,  # Synthesis: no thinking needed, write the answer directly
        )
        thinking_chunks: list[str] = []
        content_chunks: list[str] = []
        try:
            async for chunk in self._gateway.stream(request):
                if getattr(chunk, "is_thinking", False):
                    thinking_chunks.append(str(chunk))
                    accum_thinking = "".join(thinking_chunks)
                    self._emit_event(
                        "thinking_token",
                        {
                            "token": str(chunk),
                            "accumulated": accum_thinking,
                            "task_id": state.task_id,
                        },
                    )
                else:
                    content_chunks.append(str(chunk))
                    accum = "".join(content_chunks)
                    if "</think>" in accum:
                        clean_accum = re.sub(r"^.*?<\/think>", "", accum, flags=re.DOTALL).lstrip()
                        if clean_accum:
                            self._emit_event(
                                "token",
                                {
                                    "token": str(chunk),
                                    "accumulated": clean_accum,
                                    "task_id": state.task_id,
                                },
                            )
                    elif "<think>" not in accum and not accum.startswith("<think>"):
                        self._emit_event(
                            "token",
                            {
                                "token": str(chunk),
                                "accumulated": accum,
                                "task_id": state.task_id,
                            },
                        )
        except Exception as stream_err:
            _logger.warning("agent.synthesize_stream_fallback  task_id=%s  err=%s", state.task_id, stream_err)
            if not content_chunks:
                response = await self._gateway.generate(request)
                content_chunks.append(response.content)
                if response.raw and isinstance(response.raw, dict):
                    resp_msg = response.raw.get("message", {})
                    if isinstance(resp_msg, dict) and resp_msg.get("thinking"):
                        thinking_chunks.append(resp_msg["thinking"])

        raw_content = "".join(content_chunks)
        self._emit_event("generating_done", {"model": state.selected_model, "task_id": state.task_id})
        thinking_text = "".join(thinking_chunks).strip() if thinking_chunks else None
        if not thinking_text and "<think>" in raw_content:
            think_match = re.search(r"<think>(.*?)(?:<\/think>|$)", raw_content, flags=re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()

        content = raw_content
        if content:
            if "</think>" in content:
                content = re.sub(r"^.*?<\/think>", "", content, flags=re.DOTALL).strip()
            elif "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            state.messages.append(
                AgentMessage(role="assistant", content=content, thinking=thinking_text)
            )
        return state

    def _make_tool_call(self, state: AgentState, step: PlanStep) -> ToolCall:
        """Create a ToolCall from a PlanStep."""
        return ToolCall(
            call_id=f"{state.task_id}-call-{len(state.tool_calls)}",
            tool_name=step.tool_name or "",
            arguments=dict(step.inputs),
            phase=ExecutionPhase.TOOL_REQUEST,
            step_index=state.current_step,
        )

    def _policy_blocked(self, state: AgentState, call: ToolCall) -> bool:
        """Evaluate ``call`` against the process policy engine.

        Runs **before** the tool executor is invoked: no tool/provider/sandbox
        operation happens until the policy engine has said ALLOW.  When the
        decision is DENY or REQUIRE_APPROVAL the step is recorded as an
        observation (a safe, non-sensitive string) and the step is advanced so
        the agent can move on; ``True`` is returned so the caller continues the
        loop without executing the tool.

        Returns:
            ``True`` if the tool call was blocked by policy (do NOT execute).
            ``False`` if the agent may proceed to execution.  When no policy
            engine has been installed the agent behaves as before Phase 6 —
            policy checks are skipped entirely (backward compatibility).
        """
        policy_engine = get_policy_engine()
        if policy_engine is None:
            return False

        decision = policy_engine.evaluate(call)

        # Log policy decision to audit logger
        try:
            audit_logger = get_audit_logger()
            audit_logger.log_policy_decision(
                capability=decision.capability or "unknown",
                action=decision.action or "unknown",
                decision=decision.decision,
                reason=decision.reason,
                correlation_id=call.call_id,
            )
        except Exception:
            # Ignore audit logging errors to prevent breaking the agent
            pass

        if decision.is_allowed():
            return False

        if decision.requires_approval():
            _logger.info(
                "agent.tool_policy  task_id=%s  tool=%s  decision=require_approval",
                state.task_id,
                call.tool_name,
            )
            state.observations.append(
                Observation(
                    call_id=call.call_id,
                    content=f"[Approval Required] {decision.reason}",
                    source="policy_engine",
                )
            )
        else:
            _logger.info(
                "agent.tool_policy  task_id=%s  tool=%s  decision=deny",
                state.task_id,
                call.tool_name,
            )
            state.observations.append(
                Observation(
                    call_id=call.call_id,
                    content=f"[Policy Denied] {decision.reason}",
                    source="policy_engine",
                )
            )

        state.current_step += 1
        state.completed_steps += 1
        return True

    def _check_repetitive(self, state: AgentState, call: ToolCall) -> None:
        """Check for repetitive tool calls and raise if threshold exceeded."""
        if not call.tool_name:
            return
        # Count consecutive identical tool+args from the most recent calls.
        window = self._config.max_repetitive_tool_calls
        # Note: ``call`` is already in ``state.tool_calls`` (appended before
        # this check), so we look at the most recent ``window`` calls.
        recent = state.tool_calls[-window:]
        if len(recent) < window:
            return
        # Compare each call by (tool_name, frozenset of argument items) so
        # dicts of the same content compare equal regardless of insertion
        # order. We must NOT include raw argument values in any error
        # message; the comparison is in-memory only.
        names = [c.tool_name for c in recent]
        arg_keysets = [frozenset(c.arguments.items()) for c in recent]
        target_keyset = frozenset(call.arguments.items())
        if len(set(names)) == 1 and all(ak == target_keyset for ak in arg_keysets):
            raise RepetitiveToolCallError(
                f"Tool {call.tool_name!r} called {window} times with identical arguments. "
                f"Aborting to prevent infinite loop."
            )

    async def _verify(self, state: AgentState) -> AgentState:
        """Call the verifier and record the result."""
        self._emit_event("verify_start", {"task_id": state.task_id})
        try:
            result = await self._verifier.verify(state)
        except Exception as exc:
            _logger.warning(
                "agent.verify_error  task_id=%s  error=%s",
                state.task_id,
                type(exc).__name__,
            )
            result = type("VerificationResult", (), {
                "passed": False,
                "reason": f"Verifier raised: {exc}",
                "suggestions": (),
            })()  # type: ignore[return-value]

        state.verification_result = result
        self._emit_event("verify_done", {"passed": result.passed, "reason": getattr(result, "reason", None), "task_id": state.task_id})
        if not result.passed:
            _logger.warning(
                "agent.verify_fail  task_id=%s  reason=%s",
                state.task_id,
                result.reason,
            )
        else:
            _logger.info(
                "agent.verify_pass  task_id=%s",
                state.task_id,
            )
        return state