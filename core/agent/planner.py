"""Built-in planners for the Agent Runtime.

This module provides simple, production-usable planners. They are deliberately
conservative — they do not require an LLM.

* :class:`SimplePlanner` — produces a single-step plan for the task using
  keyword matching on the task string. Intended as a baseline that always
  produces *some* plan.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

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
        "document": "create_document",
        "docx": "create_document",
        "report": "create_document",
        "search": "search_knowledge",
        "find": "search_knowledge",
        "knowledge": "search_knowledge",
        "create document": "create_document",
        "create docx": "create_document",
        "artifact": None,
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

        The plan is always either tool steps or an empty plan (no tools).
        When the state carries validated attachment names and a read/summary is
        requested, a ``read_file`` step is planned per attachment so the tool
        receives the correct workspace-relative ``path``.
        """
        task_lower = task.lower()

        # Attachments in the task state are authoritative:
        # If an image attachment is provided, plan vision analysis.
        # If a document attachment is provided, plan read_file.
        if state.attachments:
            image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
            image_attachments = [a for a in state.attachments if a.lower().endswith(image_exts)]
            doc_attachments = [a for a in state.attachments if not a.lower().endswith(image_exts)]

            # 1. Image attachments -> Vision tool ONLY for image files
            if image_attachments and ("analyze_image" in available_tools or "ocr_image" in available_tools):
                is_ocr = any(k in task_lower for k in ("ocr", "transcribe", "extract text", "read text"))
                if is_ocr and "ocr_image" in available_tools:
                    vision_tool = "ocr_image"
                else:
                    vision_tool = "analyze_image" if "analyze_image" in available_tools else "ocr_image"
                steps = tuple(
                    PlanStep(
                        step_id=f"step-{i}",
                        description=f"Analyze attached image {name}",
                        tool_name=vision_tool,
                        inputs={"path": name, "prompt": task} if vision_tool == "analyze_image" else {"path": name},
                        reason=f"Task references attached image {name!r}",
                    )
                    for i, name in enumerate(image_attachments)
                )
                _logger.info(
                    "planner: %s planned for %d image attachment(s)",
                    vision_tool,
                    len(steps),
                )
                return Plan(
                    goal=task,
                    steps=steps,
                    reasoning="Task references attached image; planned vision analysis.",
                )

            # 2. Document attachments -> read_file ONLY for non-image files
            if doc_attachments and "read_file" in available_tools:
                steps = tuple(
                    PlanStep(
                        step_id=f"step-{i}",
                        description=f"Read attached file {name}",
                        tool_name="read_file",
                        inputs={"path": name},
                        reason=f"Task references attached file {name!r}",
                    )
                    for i, name in enumerate(doc_attachments)
                )
                _logger.info(
                    "planner: read_file planned for %d document attachment(s)",
                    len(steps),
                )
                return Plan(
                    goal=task,
                    steps=steps,
                    reasoning=(
                        "Task references attached document; planned read_file per attachment."
                    ),
                )

        # Check if task text references an image file directly
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
        image_match = re.search(
            r'["\']?([a-zA-Z0-9_\-\.\/\\]+\.(?:png|jpg|jpeg|webp|bmp|tiff))["\']?',
            task,
            re.IGNORECASE,
        )
        if image_match and ("analyze_image" in available_tools or "ocr_image" in available_tools):
            img_path = image_match.group(1).strip()
            is_ocr = any(k in task_lower for k in ("ocr", "transcribe", "extract text", "read text"))
            if is_ocr and "ocr_image" in available_tools:
                vision_tool = "ocr_image"
            else:
                vision_tool = "analyze_image" if "analyze_image" in available_tools else "ocr_image"
            step = PlanStep(
                step_id="step-0",
                description=f"Analyze image {img_path}",
                tool_name=vision_tool,
                inputs={"path": img_path, "prompt": task} if vision_tool == "analyze_image" else {"path": img_path},
                reason=f"Task references image {img_path!r}",
            )
            _logger.info("planner: %s planned for prompt-referenced image %s", vision_tool, img_path)
            return Plan(
                goal=task,
                steps=(step,),
                reasoning="Task references image; planned vision analysis.",
            )

        # Check if task text references a document or code file directly
        doc_match = re.search(
            r'["\']?([a-zA-Z0-9_\-\.\/\\]+\.(?:txt|md|json|csv|py|cpp|c|h|log|yaml|yml|xml|html|js|ts))["\']?',
            task,
            re.IGNORECASE,
        )
        if doc_match and "read_file" in available_tools:
            doc_path = doc_match.group(1).strip()
            step = PlanStep(
                step_id="step-0",
                description=f"Read file {doc_path}",
                tool_name="read_file",
                inputs={"path": doc_path},
                reason=f"Task references document file {doc_path!r}",
            )
            _logger.info("planner: read_file planned for prompt-referenced file %s", doc_path)
            return Plan(
                goal=task,
                steps=(step,),
                reasoning="Task references document file; planned read_file.",
            )

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
                inputs = self._extract_inputs(task, tool_name, state=state)
                if tool_name == "read_file" and not inputs.get("path"):
                    _logger.warning("planner: read_file requested without path; falling back to direct answer")
                    return Plan(
                        goal=task,
                        steps=(),
                        reasoning="No file path or attachment was specified for read_file.",
                    )

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

    def _extract_inputs(
        self, task: str, tool_name: str, state: Optional[AgentState] = None
    ) -> dict[str, object]:
        """Extract arguments for the tool from the task string."""
        if tool_name == "read_file":
            if state is not None and state.attachments:
                image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
                doc_atts = [a for a in state.attachments if not a.lower().endswith(image_exts)]
                if doc_atts:
                    return {"path": doc_atts[0], "task": task}
            match = re.search(r'["\'](.+?)["\']|([^\s]+(?:\.[a-zA-Z0-9]+)+)', task)
            if match:
                filename = match.group(1) or match.group(2)
                return {"path": filename, "task": task}
            return {"task": task}

        if tool_name == "search_knowledge":
            return {"query": task}

        if tool_name == "create_document":
            match = re.search(r'["\'](.+?\.docx)["\']|([^\s]+\.docx)', task, re.IGNORECASE)
            path = "document.docx"
            if match:
                path = match.group(1) or match.group(2)
            title_match = re.search(r'title\s+["\']?([^"\'\n,]+)["\']?', task, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Generated Document"
            return {"path": path, "title": title, "content": task}

        if tool_name == "create_artifact":
            return {"content": task, "artifact_type": "text"}

        if tool_name == "execute_code":
            return {"code": "", "language": "python", "task": task}

        # Default: pass the whole task.
        return {"task": task}

    async def update_plan(
        self,
        plan: Plan,
        step_index: int,
        state: AgentState,
    ) -> Plan:
        """Update the plan after a step has been executed."""
        if step_index >= len(plan.steps):
            return plan

        remaining_steps = plan.steps[step_index + 1 :]
        return Plan(
            goal=plan.goal,
            steps=remaining_steps,
            reasoning=plan.reasoning,
        )
