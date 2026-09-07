"""Deterministic, capability-based Model Router.

The router's job is to answer the question **"which configured model should
handle this task?"** — and only that. It does not call any provider, never
talks to Ollama or any other backend, and never uses an LLM to make a
routing decision. The output is a :class:`RoutingDecision` that the caller
hands to a :class:`core.llm.ModelGateway` for actual generation.

Design principles
-----------------

* **Deterministic.** Identical input + identical registry → identical decision.
  All tie-breaks are explicit and ordered.
* **Explainable.** The decision includes a free-text ``reason`` and the score
  components used to rank candidates.
* **No silent downgrades.** A vision task is not silently routed to a
  text-only model. A coding task is not silently routed to a model that
  lacks the coding capability. When the request cannot be satisfied, the
  router raises :class:`NoSuitableModelError` with a clear reason.
* **No LLM, no ML, no prompt tricks.** Scoring is a small, fixed integer
  formula over declared capabilities, modalities, enabled status, and
  priority.

Scoring
-------

For each enabled model that is not in the request's exclusion list, the
router computes a score as::

    score = 0
    if all required capabilities are present:   score += 100
    score += 2 * (number of required capabilities matched)
    if all required input modalities are present: score += 50
    if model has extra useful capabilities:       score += 1 per cap
    score += model.priority

A model that fails any required-capability check, or fails any required
input modality, scores -1 and is excluded from consideration. The highest
score wins; ties break by ``logical_name`` lexicographic order to keep
results stable across runs.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.routing.errors import NoSuitableModelError
from core.routing.registry import ModelRegistry
from core.routing.types import (
    Capability,
    Modality,
    ModelDefinition,
    RoutingDecision,
    RoutingRequest,
    TaskType,
)


_logger = logging.getLogger("sovereign-ai.router")

# A model that fails any hard requirement is excluded outright.
_DISQUALIFIED = -1

# Hard-requirement weights.
_REQUIRED_CAPABILITIES_PRESENT = 100
_REQUIRED_MODALITIES_PRESENT = 50

# Soft-requirement weights.
_PER_REQUIRED_CAPABILITY_MATCH = 2
_PER_EXTRA_CAPABILITY = 1


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Maps a high-level :class:`TaskType` to the capabilities a model must
#: have to handle it. The router uses this when the request does not
#: specify ``required_capabilities`` explicitly.
TASK_TYPE_DEFAULT_CAPABILITIES: dict[TaskType, frozenset[Capability]] = {
    TaskType.CHAT: frozenset({Capability.GENERAL}),
    TaskType.CODING: frozenset({Capability.CODING}),
    TaskType.VISION: frozenset({Capability.VISION}),
    TaskType.DOCUMENT: frozenset({Capability.DOCUMENT_ANALYSIS}),
    TaskType.REASONING: frozenset({Capability.REASONING}),
}

#: Maps a high-level :class:`TaskType` to the input modalities the model
#: must support. A vision task defaults to ``{TEXT, IMAGE}``; everything
#: else defaults to ``{TEXT}``. The router uses this when the request
#: does not specify ``input_modalities`` explicitly.
TASK_TYPE_DEFAULT_INPUT_MODALITIES: dict[TaskType, frozenset[Modality]] = {
    TaskType.CHAT: frozenset({Modality.TEXT}),
    TaskType.CODING: frozenset({Modality.TEXT}),
    TaskType.VISION: frozenset({Modality.TEXT, Modality.IMAGE}),
    TaskType.DOCUMENT: frozenset({Modality.TEXT}),
    TaskType.REASONING: frozenset({Modality.TEXT}),
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ModelRouter:
    """A deterministic, capability-based Model Router.

    The router is constructed with a :class:`ModelRegistry` and is safe for
    concurrent use. It performs no I/O: routing is a pure function of the
    request and the registry contents.

    Example::

        router = ModelRouter(registry=my_registry)
        decision = router.route(RoutingRequest(task_type=TaskType.CODING))
        # decision.model.logical_name == "coding"
        # decision.reason  == "Matched required capabilities {coding}; ..."
    """

    def __init__(self, registry: ModelRegistry) -> None:
        if not isinstance(registry, ModelRegistry):
            raise TypeError(
                f"ModelRouter requires a ModelRegistry, got {type(registry).__name__}"
            )
        if len(registry) == 0:
            raise ValueError(
                "ModelRouter requires a non-empty ModelRegistry; "
                "build one with ModelRegistry(...) or load_default_registry()."
            )
        self._registry = registry

    @property
    def registry(self) -> ModelRegistry:
        """The registry this router consults. Read-only by convention."""
        return self._registry

    # ------------------------------------------------------------------ Route

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Select a model for the given :class:`RoutingRequest`.

        Returns:
            RoutingDecision: A typed decision object describing the chosen
            model, the reason for the choice, and the deterministic score.

        Raises:
            NoSuitableModelError: If no configured model satisfies the
                request. The exception carries a diagnostic ``reason``
                string safe to surface to end users.
        """
        if not isinstance(request, RoutingRequest):
            raise TypeError(
                f"ModelRouter.route requires a RoutingRequest, "
                f"got {type(request).__name__}"
            )

        required_caps = self._required_capabilities(request)
        required_modalities = self._required_input_modalities(request)

        # 1. Preferred-model fast path: honour preferred_model only if it
        #    actually satisfies the request. This is a hard precondition,
        #    not a soft preference — we do not silently fall back to a model
        #    that the caller asked for if it cannot do the work.
        if request.preferred_model is not None:
            decision = self._try_preferred(request, required_caps, required_modalities)
            if decision is not None:
                _logger.info(
                    "route.decision  task=%s  preferred=%s  chosen=%s  score=%d  reason=%s",
                    request.task_type.value,
                    request.preferred_model,
                    decision.model.logical_name,
                    decision.score,
                    decision.reason,
                )
                return decision
            # Preferred model failed; fall through to normal selection so
            # the caller still gets a usable model rather than an error.
            # We log this so the rejection is auditable.
            _logger.info(
                "route.preferred_rejected  task=%s  preferred=%s  falling_back_to=capability_match",
                request.task_type.value,
                request.preferred_model,
            )

        # 2. Normal capability-matched selection.
        candidates = self._rank_candidates(request, required_caps, required_modalities)
        if not candidates:
            self._raise_no_suitable(request, required_caps, required_modalities)

        best_model, best_score, reason, matched, modality_ok = candidates[0]
        decision = RoutingDecision(
            model=best_model,
            reason=reason,
            score=best_score,
            matched_capabilities=matched,
            modality_satisfied=modality_ok,
            preferred_honoured=False,
        )
        _logger.info(
            "route.decision  task=%s  chosen=%s  score=%d  reason=%s",
            request.task_type.value,
            decision.model.logical_name,
            decision.score,
            decision.reason,
        )
        return decision

    # ----------------------------------------------------------------- Helpers

    def _required_capabilities(self, request: RoutingRequest) -> frozenset[Capability]:
        if request.required_capabilities:
            return request.required_capabilities
        return TASK_TYPE_DEFAULT_CAPABILITIES.get(
            request.task_type, frozenset({Capability.GENERAL})
        )

    def _required_input_modalities(self, request: RoutingRequest) -> frozenset[Modality]:
        if request.input_modalities:
            return request.input_modalities
        return TASK_TYPE_DEFAULT_INPUT_MODALITIES.get(
            request.task_type, frozenset({Modality.TEXT})
        )

    def _try_preferred(
        self,
        request: RoutingRequest,
        required_caps: frozenset[Capability],
        required_modalities: frozenset[Modality],
    ) -> Optional[RoutingDecision]:
        """Try to honour the request's ``preferred_model``.

        Returns ``None`` if the preferred model is unknown, disabled, or
        fails any hard requirement. In that case the router falls back to
        normal capability selection.
        """
        if not self._registry.has(request.preferred_model):  # type: ignore[arg-type]
            return None
        model = self._registry.get(request.preferred_model)  # type: ignore[arg-type]
        if not model.enabled:
            return None
        if model.logical_name in request.excluded_models:
            return None

        missing_caps = required_caps - model.capabilities
        if missing_caps:
            # Preferred model lacks a required capability — refuse to honour
            # the preference. This is the explicit guarantee in the spec:
            # "Honor preferred_model only if it satisfies the requested
            # capabilities and is enabled."
            return None

        missing_modalities = required_modalities - model.input_modalities
        if missing_modalities:
            return None

        # Preferred model passes; build a decision with a stable score.
        score = (
            _REQUIRED_CAPABILITIES_PRESENT
            + _REQUIRED_MODALITIES_PRESENT
            + 2 * len(required_caps & model.capabilities)
            + model.priority
        )
        reason = (
            f"Preferred model '{model.logical_name}' satisfies all required "
            f"capabilities {sorted(c.value for c in required_caps)} and "
            f"input modalities {sorted(m.value for m in required_modalities)}"
        )
        return RoutingDecision(
            model=model,
            reason=reason,
            score=score,
            matched_capabilities=required_caps & model.capabilities,
            modality_satisfied=True,
            preferred_honoured=True,
        )

    def _rank_candidates(
        self,
        request: RoutingRequest,
        required_caps: frozenset[Capability],
        required_modalities: frozenset[Modality],
    ) -> list[tuple[ModelDefinition, int, str, frozenset[Capability], bool]]:
        """Score every eligible model and return them in deterministic order."""
        scored: list[tuple[ModelDefinition, int, str, frozenset[Capability], bool]] = []
        for model in self._registry.all():
            if not model.enabled:
                continue
            if model.logical_name in request.excluded_models:
                continue

            missing_caps = required_caps - model.capabilities
            missing_modalities = required_modalities - model.input_modalities

            if missing_caps:
                # Hard fail. A vision task must not be silently routed to a
                # text-only model, and a coding task must not be silently
                # routed to a model that lacks the coding capability.
                continue
            if missing_modalities:
                continue

            matched = required_caps & model.capabilities
            modality_ok = required_modalities.issubset(model.input_modalities)
            extra_caps = model.capabilities - required_caps

            score = (
                _REQUIRED_CAPABILITIES_PRESENT
                + _REQUIRED_MODALITIES_PRESENT
                + _PER_REQUIRED_CAPABILITY_MATCH * len(matched)
                + _PER_EXTRA_CAPABILITY * len(extra_caps)
                + model.priority
            )

            reason = self._build_reason(
                model=model,
                required_caps=required_caps,
                required_modalities=required_modalities,
                matched=matched,
                modality_ok=modality_ok,
            )
            scored.append((model, score, reason, matched, modality_ok))

        # Sort: highest score first; on tie, lexicographic logical_name.
        scored.sort(key=lambda item: (-item[1], item[0].logical_name))
        return scored

    def _build_reason(
        self,
        model: ModelDefinition,
        required_caps: frozenset[Capability],
        required_modalities: frozenset[Modality],
        matched: frozenset[Capability],
        modality_ok: bool,
    ) -> str:
        cap_list = ", ".join(sorted(c.value for c in required_caps)) or "<none>"
        mod_list = ", ".join(sorted(m.value for m in required_modalities)) or "<none>"
        matched_list = ", ".join(sorted(c.value for c in matched)) or "<none>"
        return (
            f"Selected '{model.logical_name}' (provider={model.provider}, "
            f"model={model.provider_model}, priority={model.priority}); "
            f"required capabilities [{cap_list}] all present; "
            f"matched=[{matched_list}]; "
            f"input modalities [{mod_list}] satisfied={modality_ok}"
        )

    def _raise_no_suitable(
        self,
        request: RoutingRequest,
        required_caps: frozenset[Capability],
        required_modalities: frozenset[Modality],
    ) -> None:
        enabled_names = [m.logical_name for m in self._registry.enabled()]
        if not enabled_names:
            reason = "no enabled models are registered"
        else:
            reason = (
                f"no enabled model matches required capabilities "
                f"{sorted(c.value for c in required_caps)} and input modalities "
                f"{sorted(m.value for m in required_modalities)}; "
                f"enabled models: {enabled_names}"
            )
        _logger.warning(
            "route.no_suitable  task=%s  required_caps=%s  modalities=%s",
            request.task_type.value,
            sorted(c.value for c in required_caps),
            sorted(m.value for m in required_modalities),
        )
        raise NoSuitableModelError(
            task_type=request.task_type,
            required_capabilities=set(required_caps),
            input_modalities=set(required_modalities),
            reason=reason,
        )
