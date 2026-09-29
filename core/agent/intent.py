"""Deterministic, instantaneous intent classification for SovereignAI Workbench.

Classifies incoming tasks into:
- FAST: greetings, thanks, goodbye, acknowledgments, trivial system queries (no LLM, <0.1ms)
- SIMPLE: short factual questions and simple explanations not requiring tools
- AGENT: multi-step, tool-using, document, vision, code, RAG, or policy-controlled tasks

CRITICAL SAFETY INVARIANT:
Never bypass agent or security checks if attachments, tool keywords, or
security-relevant patterns are present.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Sequence, Tuple


class TaskIntent(str, Enum):
    FAST = "fast"
    SIMPLE = "simple"
    AGENT = "agent"


# Pre-compiled regex patterns for deterministic FAST classification
_GREETING_PATTERNS = re.compile(
    r"^(?:hello|hi|hey|hi\s+there|hello\s+there|good\s+morning|good\s+afternoon|good\s+evening|greetings|howdy|hiya|yo)[\s!.,?]*$",
    re.IGNORECASE,
)

_THANKS_PATTERNS = re.compile(
    r"^(?:thanks|thank\s+you|thank\s+you\s+so\s+much|thanks\s+a\s+lot|thx|appreciate\s+it|many\s+thanks)[\s!.,?]*$",
    re.IGNORECASE,
)

_GOODBYE_PATTERNS = re.compile(
    r"^(?:bye|goodbye|bye\s+bye|see\s+you|see\s+ya|cya|farewell|have\s+a\s+good\s+day|good\s+night)[\s!.,?]*$",
    re.IGNORECASE,
)

_ACK_PATTERNS = re.compile(
    r"^(?:ok|okay|cool|got\s+it|understood|sure|sounds\s+good|great|nice|alright|perfect)[\s!.,?]*$",
    re.IGNORECASE,
)

_SYSTEM_INFO_PATTERNS = re.compile(
    r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help|ping|test)[\s!.,?]*$",
    re.IGNORECASE,
)

# Tool / Action keywords that MUST always go to AGENT or at least never FAST
# Using word boundaries to avoid false positives (e.g. 'rm' in 'terms')
_AGENT_TOOL_PATTERN = re.compile(
    r"\b(?:read\s+file|read|write\s+file|write|delete|list\s+files|file|files|"
    r"document|docx|report|search\s+knowledge|search|knowledge|rag|"
    r"create\s+document|create\s+docx|execute\s+code|code|calculate|compute|"
    r"run\s+code|run\s+script|python|script|ocr|ocr\s+image|image|images|"
    r"picture|photo|png|jpg|jpeg|pdf|txt|eval|sandbox|rm|cat|bash|shell)\b",
    re.IGNORECASE,
)

# Complex reasoning / multi-step patterns
_COMPLEX_PATTERNS = re.compile(
    r"\b(?:compare\s+and\s+contrast|pros\s+and\s+cons|detailed\s+report|"
    r"step\s+by\s+step|solve\s+all|bullet\s+points|comprehensive)\b",
    re.IGNORECASE,
)


def classify_intent(
    task: str,
    attachments: Optional[Sequence[str]] = None,
) -> Tuple[TaskIntent, Optional[str]]:
    """Classify user intent deterministically and instantaneously (<0.1ms).

    Args:
        task: Raw task prompt string.
        attachments: Optional sequence of workspace-relative attachments.

    Returns:
        (TaskIntent, fast_path_response_or_none)
    """
    clean_task = task.strip()
    if not clean_task:
        return TaskIntent.FAST, "Hello! How can I help you today?"

    # 1. Any attachment MUST be handled by AGENT runtime for security and tooling
    if attachments and len(attachments) > 0:
        return TaskIntent.AGENT, None

    task_lower = clean_task.lower()

    # 2. Check for tool keywords or code/file syntax -> AGENT
    if _AGENT_TOOL_PATTERN.search(task_lower):
        return TaskIntent.AGENT, None

    # Numbered multi-step questions -> AGENT
    if any(f"{i}." in task_lower for i in range(1, 10)) or _COMPLEX_PATTERNS.search(task_lower):
        return TaskIntent.AGENT, None

    # 3. Deterministic FAST path matching (strictly for short, conversational turns)
    if len(clean_task) <= 60:
        if _GREETING_PATTERNS.match(clean_task):
            return TaskIntent.FAST, "Hello! How can I help you today?"
        if _THANKS_PATTERNS.match(clean_task):
            return TaskIntent.FAST, "You're very welcome! Let me know if you need assistance with anything else."
        if _GOODBYE_PATTERNS.match(clean_task):
            return TaskIntent.FAST, "Goodbye! Have a great day!"
        if _ACK_PATTERNS.match(clean_task):
            return TaskIntent.FAST, "Understood! Let me know what you'd like to do next."
        if _SYSTEM_INFO_PATTERNS.match(clean_task):
            if "ping" in task_lower or "test" in task_lower:
                return TaskIntent.FAST, "Pong! SovereignAI Workbench is online and ready."
            return (
                TaskIntent.FAST,
                "I am SovereignAI Workbench, an air-gapped, sovereign multimodal agent. "
                "I can analyze files, inspect documents and images, run sandboxed code, "
                "and assist with technical tasks entirely on-premise."
            )

    # 4. Check for SIMPLE queries: short factual questions / conceptual queries without tools
    if len(clean_task) <= 150:
        # Starts with common question words or explanation requests
        simple_prefixes = (
            "what is", "what are", "how does", "explain", "describe",
            "define", "why does", "tell me about", "briefly explain",
            "meaning of", "difference between"
        )
        if any(task_lower.startswith(p) for p in simple_prefixes) or clean_task.endswith("?"):
            return TaskIntent.SIMPLE, None

    # Default fallback: if longer or unstructured, route to AGENT
    return TaskIntent.AGENT, None
