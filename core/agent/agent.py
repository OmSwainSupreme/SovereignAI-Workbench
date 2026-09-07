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

import inspect
import logging
import time
import uuid
from typing import Optional, Sequence

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
        tool_executor: ToolExecutor,
        planner: Planner,
        verifier: Verifier,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        if model_router is None:
            raise TypeError("Agent requires a ModelRouter")
        if tool_executor is None:
            raise TypeError("Agent requires a ToolExecutor")
        if planner is None:
            raise TypeError("Agent requires a Planner")
        if verifier is None:
            raise TypeError("Agent requires a Verifier")

        self._router = model_router
        self._executor = tool_executor
        self._planner = planner
        self._verifier = verifier
        self._config = config or AgentConfig()
        # The tool registry is optional: if not given, the agent falls back
        # to reading tool names from the executor (if it has a ``registry``
        # attribute, as SyncToolExecutor does).
        self._tool_registry = tool_registry

    # ------------------------------------------------------------------ Run

    async def run(self, task: str, task_id: Optional[str] = None) -> AgentResult:
        """Run the agent on the given task.

        This is the main entry point. The agent moves through its state machine
        and returns an :class:`AgentResult` when done.

        Args:
            task: The user's task string. Must be non-empty.
            task_id: Optional unique identifier for this run. A UUID is generated
                if not provided.

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
        state = AgentState(task_id=task_id, task=task)
        state.status = AgentStatus.RUNNING
        state.started_at = time.monotonic()

        _logger.info(
            "agent.start  task_id=%s  task_length=%d  max_iterations=%d  max_tool_calls=%d",
            task_id,
            len(task),
            self._config.max_iterations,
            self._config.max_tool_calls,
        )

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
            return AgentResult.from_state(state)
        except AgentTimeoutError:
            state.status = AgentStatus.TIMEOUT
            state.current_phase = ExecutionPhase.FAILED
            _logger.warning("agent.timeout  task_id=%s  iterations=%d", task_id, state.iteration)
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
            return AgentResult.from_state(state, error=type(exc).__name__)

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
            try:
                tool_result = await self._executor.execute(tool_call)
            except UnknownToolError as exc:
                # Planner asked for a non-existent tool — this is a planning error.
                state.errors.append(f"Unknown tool: {exc.tool_name}")
                raise ToolExecutionError(
                    tool_name=exc.tool_name,
                    message="Tool referenced by plan is not registered",
                    cause=exc,
                ) from exc

            # Log tool execution end
            try:
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

        # --- COMPLETE ---
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
        return state

    async def _route(self, state: AgentState) -> str:
        """Route to select a model for the current task.

        The agent derives a :class:`RoutingRequest` from the task and plan,
        asks the ModelRouter, and returns the selected logical model name.
        """
        # For the initial implementation, we use CHAT as the task type.
        # Future agents can do more sophisticated task type detection.
        request = RoutingRequest(
            task_type=TaskType.CHAT,
            required_capabilities=frozenset(),
            input_modalities=frozenset(),
        )
        try:
            route_result = self._router.route(request)
            if inspect.isawaitable(route_result):
                decision = await route_result
            else:
                decision = route_result
            return decision.model.logical_name
        except NoSuitableModelError as exc:
            raise RoutingFailureError(
                f"ModelRouter could not select a model: {exc.reason}",
                cause=exc,
            ) from exc
        except RoutingError as exc:
            raise RoutingFailureError(
                f"ModelRouter error: {exc}",
                cause=exc,
            ) from exc

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
