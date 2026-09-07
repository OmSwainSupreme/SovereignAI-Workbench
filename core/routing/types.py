"""Framework-agnostic data types for the model router.

These types are deliberately plain dataclasses — they must not depend on
FastAPI, Pydantic, or any other framework. The application layer
(``backend/app``) may wrap them in Pydantic DTOs for HTTP transport.

This mirrors the design convention of :mod:`core.llm.types`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Capability(str, Enum):
    """A discrete capability that a model may possess.

    Capabilities are task-oriented, not provider-specific. A model's set of
    capabilities is declared in its configuration; the router matches the
    request's required capabilities against this set.
    """

    #: General-purpose chat, Q&A, and reasoning.
    GENERAL = "general"
    #: Code generation, explanation, refactoring.
    CODING = "coding"
    #: Image / visual understanding (captioning, classification, OCR-free analysis).
    VISION = "vision"
    #: Document analysis and summarisation over long context.
    DOCUMENT_ANALYSIS = "document_analysis"
    #: Long-form reasoning / chain-of-thought.
    REASONING = "reasoning"
    #: Tool / function calling.
    TOOL_USE = "tool_use"


class Modality(str, Enum):
    """A modality that a model can accept as input or produce as output."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


class TaskType(str, Enum):
    """The high-level category of work the caller is asking the router to do.

    This is a routing hint, not a strict contract. The router uses the task
    type together with the explicit ``required_capabilities`` set to pick a
    model. If ``required_capabilities`` is empty, the router derives a
    sensible default set from the task type.
    """

    #: General chat / Q&A.
    CHAT = "chat"
    #: Code generation or refactoring.
    CODING = "coding"
    #: Image understanding, captioning, visual analysis.
    VISION = "vision"
    #: Document analysis or summarisation.
    DOCUMENT = "document"
    #: Reasoning / analysis that does not fit a more specific category.
    REASONING = "reasoning"


# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelDefinition:
    """A single configured logical model.

    A logical model is the *router-facing* identity; the actual provider may
    call it something different (``provider_model``). New models are added
    by configuration — the router itself does not need to change.

    Attributes:
        logical_name: Stable, human-readable identifier (e.g. ``"coding"``).
        provider: Name of the provider this model is served by
            (e.g. ``"ollama"``). Must match a registered provider.
        provider_model: The provider-specific model identifier
            (e.g. ``"qwen2.5-coder:3b"``).
        capabilities: Set of capabilities the model supports.
        input_modalities: Modalities the model accepts as input.
        output_modalities: Modalities the model can produce.
        enabled: Whether the router may select this model. A disabled model
            is never returned, even if the caller asks for it by name.
        priority: Higher values win when multiple models match. Tie-breaks
            fall back to lexicographic order of ``logical_name`` for
            determinism. Negative values are allowed (e.g. for deprecated
            models that should still be routable in tests).
        context_length: Optional metadata for future context-aware routing.
            Not used by the router today; included so callers can surface it
            in audit logs and UIs.
        description: Free-form human-readable description, for diagnostics.
    """

    logical_name: str
    provider: str
    provider_model: str
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    input_modalities: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))
    output_modalities: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))
    enabled: bool = True
    priority: int = 0
    context_length: Optional[int] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Routing request / response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingRequest:
    """A request to select a model for a task.

    The router needs enough information to make a deterministic capability
    match. The structure is intentionally extensible — future fields (cost
    hints, latency budgets) can be added without breaking callers.

    Attributes:
        task_type: High-level category of work. Used to derive a default
            capability set when ``required_capabilities`` is empty.
        required_capabilities: Capabilities the chosen model must have.
            If empty, the router infers a set from ``task_type``.
        input_modalities: Modalities present in the upcoming request to the
            model (e.g. ``{TEXT}`` for chat, ``{TEXT, IMAGE}`` for an image
            + caption task). A model whose ``input_modalities`` is a
            superset of these is required when the set is non-empty.
        preferred_model: Optional logical name. Honoured only if it satisfies
            all required capabilities and is enabled. Never overrides
            capability rules.
        excluded_models: Optional logical names to skip. Useful for A/B tests
            and forced fallbacks.
    """

    task_type: TaskType = TaskType.CHAT
    required_capabilities: frozenset[Capability] = field(default_factory=frozenset)
    input_modalities: frozenset[Modality] = field(default_factory=frozenset)
    preferred_model: Optional[str] = None
    excluded_models: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Treat None as "no constraint" by normalising to empty sets / None.
        # The dataclass field defaults already do this, but if a caller passes
        # an explicit None we still want a well-formed request.
        if self.required_capabilities is None:  # type: ignore[unreachable]
            object.__setattr__(self, "required_capabilities", frozenset())
        if self.input_modalities is None:  # type: ignore[unreachable]
            object.__setattr__(self, "input_modalities", frozenset())
        if self.excluded_models is None:  # type: ignore[unreachable]
            object.__setattr__(self, "excluded_models", frozenset())


@dataclass(frozen=True)
class RoutingDecision:
    """The result of selecting a model for a task.

    This object is suitable for audit logging and UI display: every field is
    a primitive, dataclass, or enum. No Pydantic dependency.

    Attributes:
        model: The selected :class:`ModelDefinition`.
        reason: Human-readable explanation of the decision (no sensitive
            content; safe to log and surface to end users).
        score: Deterministic integer score used to rank candidates. Higher
            is better. Provided for transparency and tests, not for ML.
        matched_capabilities: Subset of the request's required capabilities
            that the selected model actually has. Normally equal to the
            request's required set, but exposed for diagnostics.
        modality_satisfied: Whether the model's input modalities cover the
            request's input modalities.
        preferred_honoured: True if the decision came from honoring the
            request's ``preferred_model``.
    """

    model: ModelDefinition
    reason: str
    score: int
    matched_capabilities: frozenset[Capability]
    modality_satisfied: bool
    preferred_honoured: bool
