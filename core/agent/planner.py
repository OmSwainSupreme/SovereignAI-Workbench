"""Built-in planners for the Agent Runtime.

This module provides simple, production-usable planners. They are deliberately
conservative — they do not require an LLM.

* :class:`SimplePlanner` — produces a single-step plan for the task using
  keyword matching on the task string. Intended as a baseline that always
  produces *some* plan.
"""
from __future__ import annotations

import logging
from typing import Sequence

from core.agent.errors import PlanningFailureError
from core.agent.interfaces import Planner
from core.agent.types import AgentState, Plan, PlanStep


_logger = logging.getLogger("sovereign-ai.agent.planner")


class SimplePlanner(Planner):
    """A rule-based planner that produces a single-step plan.

    The planner examines the task string and, if it contains known keywords,
    produces a plan that references the corresponding tool. If no keyword
    matches, it produces an empty plan (the agent will answer directly via the
    model without tools).

    This planner does NOT use an LLM. It is a baseline that demonstrates that
    the agent can work without a sophisticated planner. Future planners may
    use an LLM for multi-step decomposition.
    """

    # Keyword → tool name mapping.
    TOOL_KEYWORDS = {
        "read": "read_file",
        "file": "read_file",
        "search": "search_knowledge",
        "find": "search_knowledge",
        "knowledge": "search_knowledge",
        "create": "create_artifact",
        "write": "create_artifact",
        "artifact": "create_artifact",
        "code": "execute_code",
        "calculate": "execute_code",
        "compute": "execute_code",
        "run": "execute_code",
        "execute": "execute_code",
        "python": "execute_code",
        "fibonacci": "execute_code",
        "algorithm": "execute_code",
        "programming": "execute_code",
        "script": "execute_code",
        "explain": None,  # no tool needed — direct answer
        "what is": None,
        "how does": None,
        "describe": None,
        "summarize": None,
        "summary": None,
    }

    def __init__(self, tool_keywords: dict[str, str | None] | None = None) -> None:
        """
        Args:
            tool_keywords: Override the default keyword→tool mapping. A value
                of ``None`` means "no tool needed for this keyword".
        """
        self._keywords = tool_keywords if tool_keywords is not None else self.TOOL_KEYWORDS

    async def plan(
        self,
        task: str,
        available_tools: Sequence[str],
        state: AgentState,
    ) -> Plan:
        """Produce a plan for the task.

        The plan is always either a single tool step or an empty plan (no tools).
        """
        task_lower = task.lower()
        step: PlanStep | None = None

        for keyword, tool_name in self._keywords.items():
            if keyword in task_lower:
                if tool_name is None:
                    # Direct answer — no tool needed.
                    _logger.debug("planner: no tool needed for task %r", task[:50])
                    return Plan(goal=task, steps=(), reasoning="Direct answer; no tools needed.")

                if tool_name not in available_tools:
                    # Tool not available — fall through to direct answer.
                    _logger.warning(
                        "planner: keyword %r matched tool %r but it is not available; "
                        "falling back to direct answer",
                        keyword,
                        tool_name,
                    )
                    return Plan(
                        goal=task,
                        steps=(),
                        reasoning=f"Keyword {keyword!r} matched {tool_name!r} but tool is unavailable.",
                    )

                # Determine inputs for the tool based on the task.
                # For the initial implementation, we pass the whole task as a single argument.
                # Future planners can do better extraction.
                inputs = self._extract_inputs(task, tool_name)
                step = PlanStep(
                    step_id="step-0",
                    description=f"Call {tool_name} to help with task",
                    tool_name=tool_name,
                    inputs=inputs,
                    reason=f"Task contains keyword {keyword!r}",
                )
                break

        if step is None:
            # No keyword matched — direct answer.
            return Plan(goal=task, steps=(), reasoning="No known tool keywords found; direct answer requested.")

        return Plan(
            goal=task,
            steps=(step,),
            reasoning=f"Task contains keyword, planned single tool call: {step.tool_name}",
        )

    def _extract_inputs(self, task: str, tool_name: str) -> dict[str, object]:
        """Extract arguments for the tool from the task string.

        The initial implementation is conservative: it just passes the task
        itself as the primary argument. Subclasses or future planners can
        do better extraction with an LLM or regex.
        """
        if tool_name == "read_file":
            # Try to extract a filename from the task.
            import re

            match = re.search(r'["\'](.+?)["\']|([^\s]+(?:\.[a-zA-Z0-9]+)+)', task)
            if match:
                filename = match.group(1) or match.group(2)
                return {"path": filename, "task": task}
            return {"task": task}

        if tool_name == "search_knowledge":
            # Use the whole task as the query.
            return {"query": task}

        if tool_name == "create_artifact":
            return {"content": task, "artifact_type": "text"}

        if tool_name == "execute_code":
            # The execute_code tool takes a `code` field. The SimplePlanner
            # has no LLM; the code itself must come from the task or from
            # a heuristic. We pass the task as the `task` field (so the
            # tool's callable is exercised with structured inputs) and
            # leave `code` empty unless the task contains a clearly-delimited
            # code block. Tests that want deterministic behaviour inject
            # the code via the plan step's ``inputs`` directly; this branch
            # is for the keyword-matching path.
            return {"code": "", "language": "python", "task": task}

        # Default: pass the whole task.
        return {"task": task}

    async def update_plan(
        self,
        plan: Plan,
        step_index: int,
        state: AgentState,
    ) -> Plan:
        """Update the plan after a step has been executed.

        The default implementation removes the completed step from the plan.
        More sophisticated planners might add new steps based on observations.
        """
        if step_index >= len(plan.steps):
            return plan

        remaining_steps = plan.steps[step_index + 1 :]
        return Plan(
            goal=plan.goal,
            steps=remaining_steps,
            reasoning=plan.reasoning,
        )
