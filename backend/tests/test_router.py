"""Comprehensive unit tests for the Model Router (Phase 3).

These tests exercise the routing layer without requiring any provider backend
(Ollama, etc.) to be running. All tests use in-memory fixtures constructed
from plain Python dataclasses.

Test coverage:
- Coding task → coding model
- Document task → general model (with document_analysis capability)
- Vision task → vision model
- Disabled model is never selected
- Unsupported task (no matching model)
- Multiple matching models → priority wins
- Preferred model honoured when capable
- Preferred model rejected when it lacks required capability
- No suitable model → NoSuitableModelError
- Deterministic results for identical inputs
- RoutingDecision contains useful metadata
"""
from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import patch

import pytest

from core.routing import (
    Capability,
    Modality,
    ModelDefinition,
    ModelRouter,
    NoSuitableModelError,
    RoutingConfigurationError,
    RoutingDecision,
    RoutingRequest,
    RoutingError,
    TaskType,
)
from core.routing.errors import NoSuitableModelError as NoSuitableModelError_
from core.routing.errors import RoutingConfigurationError as RoutingConfigurationError_
from core.routing.registry import (
    ModelRegistry,
    load_registry_from_mapping,
    load_registry_from_path,
    load_default_registry,
)
from core.routing.types import Capability as Capability_
from core.routing.types import Modality as Modality_
from core.routing.types import ModelDefinition as ModelDefinition_
from core.routing.types import RoutingDecision as RoutingDecision_
from core.routing.types import RoutingRequest as RoutingRequest_
from core.routing.types import TaskType as TaskType_

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _coding_model() -> ModelDefinition:
    return ModelDefinition(
        logical_name="coding",
        provider="ollama",
        provider_model="qwen2.5-coder:3b",
        capabilities=frozenset({Capability.CODING, Capability.GENERAL}),
        input_modalities=frozenset({Modality.TEXT}),
        output_modalities=frozenset({Modality.TEXT}),
        enabled=True,
        priority=20,
    )


def _general_model() -> ModelDefinition:
    return ModelDefinition(
        logical_name="general",
        provider="ollama",
        provider_model="qwen3:4b",
        capabilities=frozenset(
            {Capability.GENERAL, Capability.REASONING, Capability.DOCUMENT_ANALYSIS}
        ),
        input_modalities=frozenset({Modality.TEXT}),
        output_modalities=frozenset({Modality.TEXT}),
        enabled=True,
        priority=10,
    )


def _vision_model() -> ModelDefinition:
    return ModelDefinition(
        logical_name="vision",
        provider="ollama",
        provider_model="qwen2.5vl:3b",
        capabilities=frozenset({Capability.VISION, Capability.GENERAL}),
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        output_modalities=frozenset({Modality.TEXT}),
        enabled=True,
        priority=20,
    )


def _low_priority_coding_model() -> ModelDefinition:
    """Same capability as _coding_model but lower priority."""
    return ModelDefinition(
        logical_name="coding-fallback",
        provider="ollama",
        provider_model="starcoder:3b",
        capabilities=frozenset({Capability.CODING, Capability.GENERAL}),
        input_modalities=frozenset({Modality.TEXT}),
        output_modalities=frozenset({Modality.TEXT}),
        enabled=True,
        priority=5,
    )


def _disabled_model() -> ModelDefinition:
    return ModelDefinition(
        logical_name="disabled-legacy",
        provider="ollama",
        provider_model="mistral:7b",
        capabilities=frozenset({Capability.GENERAL}),
        input_modalities=frozenset({Modality.TEXT}),
        output_modalities=frozenset({Modality.TEXT}),
        enabled=False,
        priority=100,
    )


def _registry(
    models: list[ModelDefinition] | None = None,
) -> ModelRegistry:
    reg = ModelRegistry()
    if models is not None:
        for m in models:
            reg.register(m)
    return reg


def _router(
    models: list[ModelDefinition] | None = None,
) -> ModelRouter:
    return ModelRouter(_registry(models))


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistryBasics:
    def test_register_and_retrieve(self) -> None:
        reg = ModelRegistry()
        model = _coding_model()
        reg.register(model)
        assert reg.has("coding")
        assert reg.get("coding") == model

    def test_register_replaces_prior_entry(self) -> None:
        reg = ModelRegistry()
        reg.register(_coding_model())
        alt = ModelDefinition(
            logical_name="coding",
            provider="ollama",
            provider_model="starcoder:3b",
        )
        reg.register(alt)
        assert reg.get("coding").provider_model == "starcoder:3b"

    def test_unregister_absent_is_noop(self) -> None:
        reg = ModelRegistry()
        reg.unregister("does-not-exist")  # must not raise

    def test_unregister_present(self) -> None:
        reg = ModelRegistry()
        reg.register(_coding_model())
        reg.unregister("coding")
        assert not reg.has("coding")

    def test_names_in_insertion_order(self) -> None:
        reg = ModelRegistry()
        reg.register(_coding_model())
        reg.register(_general_model())
        reg.register(_vision_model())
        assert reg.names() == ["coding", "general", "vision"]

    def test_all_and_enabled(self) -> None:
        reg = ModelRegistry()
        reg.register(_coding_model())
        reg.register(_disabled_model())
        assert len(reg.all()) == 2
        assert [m.logical_name for m in reg.enabled()] == ["coding"]

    def test_empty_logical_name_raises(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(RoutingConfigurationError, match="non-empty"):
            reg.register(
                ModelDefinition(
                    logical_name="",
                    provider="ollama",
                    provider_model="llama3",
                )
            )

    def test_missing_provider_raises(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(RoutingConfigurationError, match="no provider"):
            reg.register(
                ModelDefinition(
                    logical_name="bad",
                    provider="",
                    provider_model="llama3",
                )
            )

    def test_missing_provider_model_raises(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(RoutingConfigurationError, match="no provider_model"):
            reg.register(
                ModelDefinition(
                    logical_name="bad",
                    provider="ollama",
                    provider_model="",
                )
            )

    def test_contains_operator(self) -> None:
        reg = ModelRegistry()
        reg.register(_coding_model())
        assert "coding" in reg
        assert "general" not in reg

    def test_len(self) -> None:
        reg = ModelRegistry()
        assert len(reg) == 0
        reg.register(_coding_model())
        assert len(reg) == 1


class TestLoadRegistryFromMapping:
    def test_minimal_entry(self) -> None:
        data = {
            "models": [
                {
                    "logical_name": "test",
                    "provider": "ollama",
                    "provider_model": "llama3",
                }
            ]
        }
        reg = load_registry_from_mapping(data)
        assert reg.has("test")
        assert reg.get("test").provider_model == "llama3"

    def test_full_entry(self) -> None:
        data = {
            "models": [
                {
                    "logical_name": "coding",
                    "provider": "ollama",
                    "provider_model": "qwen2.5-coder:3b",
                    "capabilities": ["coding", "general"],
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "enabled": True,
                    "priority": 20,
                    "context_length": 32768,
                    "description": "Code model",
                }
            ]
        }
        reg = load_registry_from_mapping(data)
        model = reg.get("coding")
        assert Capability.CODING in model.capabilities
        assert Capability.GENERAL in model.capabilities
        assert model.priority == 20
        assert model.context_length == 32768
        assert model.description == "Code model"

    def test_invalid_capability_raises(self) -> None:
        data = {
            "models": [
                {
                    "logical_name": "bad",
                    "provider": "ollama",
                    "provider_model": "llama3",
                    "capabilities": ["not_a_capability"],
                }
            ]
        }
        with pytest.raises(RoutingConfigurationError, match="Unknown capability"):
            load_registry_from_mapping(data)

    def test_invalid_modality_raises(self) -> None:
        data = {
            "models": [
                {
                    "logical_name": "bad",
                    "provider": "ollama",
                    "provider_model": "llama3",
                    "input_modalities": ["not_a_modality"],
                }
            ]
        }
        with pytest.raises(RoutingConfigurationError, match="Unknown modality"):
            load_registry_from_mapping(data)

    def test_missing_top_level_models_key_raises(self) -> None:
        with pytest.raises(RoutingConfigurationError, match="'models'"):
            load_registry_from_mapping({})

    def test_missing_required_fields_raises(self) -> None:
        data = {"models": [{"logical_name": "bad"}]}
        with pytest.raises(RoutingConfigurationError, match="missing required"):
            load_registry_from_mapping(data)

    def test_disabled_model_in_config(self) -> None:
        data = {
            "models": [
                {
                    "logical_name": "disabled-legacy",
                    "provider": "ollama",
                    "provider_model": "mistral:7b",
                    "enabled": False,
                }
            ]
        }
        reg = load_registry_from_mapping(data)
        assert not reg.get("disabled-legacy").enabled

    def test_priority_as_int(self) -> None:
        data = {
            "models": [
                {
                    "logical_name": "high",
                    "provider": "ollama",
                    "provider_model": "llama3",
                    "priority": "30",  # string that can be int-cast
                }
            ]
        }
        reg = load_registry_from_mapping(data)
        assert reg.get("high").priority == 30


# ---------------------------------------------------------------------------
# ModelRouter construction
# ---------------------------------------------------------------------------


class TestModelRouterConstruction:
    def test_accepts_registry(self) -> None:
        reg = _registry([_coding_model()])
        router = ModelRouter(reg)
        assert router.registry is reg

    def test_rejects_non_registry(self) -> None:
        with pytest.raises(TypeError, match="ModelRegistry"):
            ModelRouter("not a registry")  # type: ignore[arg-type]

    def test_rejects_empty_registry(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ModelRouter(ModelRegistry())


# ---------------------------------------------------------------------------
# Routing: basic task types
# ---------------------------------------------------------------------------


class TestRoutingBasic:
    def test_coding_task_routes_to_coding_model(self) -> None:
        router = _router([_coding_model(), _general_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CODING))
        assert decision.model.logical_name == "coding"

    def test_coding_task_via_explicit_capability(self) -> None:
        """Request with explicit required_capabilities, no task_type."""
        router = _router([_coding_model(), _general_model()])
        decision = router.route(
            RoutingRequest(required_capabilities=frozenset({Capability.CODING}))
        )
        assert decision.model.logical_name == "coding"

    def test_document_task_routes_to_general_model(self) -> None:
        router = _router([_coding_model(), _general_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.DOCUMENT))
        assert decision.model.logical_name == "general"

    def test_vision_task_routes_to_vision_model(self) -> None:
        router = _router([_coding_model(), _vision_model(), _general_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.VISION))
        assert decision.model.logical_name == "vision"

    def test_vision_task_requires_vision_capability(self) -> None:
        """Ensure we cannot silently route vision to a non-vision model."""
        router = _router([_general_model()])  # no vision capability
        with pytest.raises(NoSuitableModelError):
            router.route(RoutingRequest(task_type=TaskType.VISION))


# ---------------------------------------------------------------------------
# Routing: disabled models
# ---------------------------------------------------------------------------


class TestRoutingDisabledModel:
    def test_disabled_model_never_selected(self) -> None:
        """Even the highest-priority disabled model is never chosen."""
        router = _router([_disabled_model(), _coding_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CHAT))
        assert decision.model.logical_name == "coding"

    def test_disabled_model_excluded_from_ranking(self) -> None:
        """Disabled model is not in the candidate list."""
        router = _router([_disabled_model(), _general_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CHAT))
        assert decision.model.logical_name == "general"


# ---------------------------------------------------------------------------
# Routing: no suitable model
# ---------------------------------------------------------------------------


class TestRoutingNoSuitableModel:
    def test_empty_registry_rejected_at_construction(self) -> None:
        """An empty registry raises ValueError immediately at router construction."""
        with pytest.raises(ValueError, match="non-empty"):
            ModelRouter(ModelRegistry())

    def test_vision_request_with_no_vision_model_raises(self) -> None:
        router = _router([_general_model()])  # no vision
        with pytest.raises(NoSuitableModelError) as exc_info:
            router.route(RoutingRequest(task_type=TaskType.VISION))
        assert "vision" in exc_info.value.reason

    def test_coding_request_with_no_coding_model_raises(self) -> None:
        router = _router([_general_model()])  # no coding
        with pytest.raises(NoSuitableModelError) as exc_info:
            router.route(RoutingRequest(task_type=TaskType.CODING))
        assert "coding" in exc_info.value.reason

    def test_exception_carries_useful_metadata(self) -> None:
        router = _router([_general_model()])
        with pytest.raises(NoSuitableModelError) as exc_info:
            router.route(RoutingRequest(task_type=TaskType.VISION))
        exc = exc_info.value
        assert exc.task_type == TaskType.VISION
        assert Capability.VISION in exc.required_capabilities
        assert exc.reason  # not empty


# ---------------------------------------------------------------------------
# Routing: priority and multiple matching models
# ---------------------------------------------------------------------------


class TestRoutingPriority:
    def test_highest_priority_wins(self) -> None:
        """When multiple models match, priority determines the winner."""
        router = _router([_general_model(), _coding_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CHAT))
        # Both have GENERAL, but coding has priority=20 > general's priority=10
        assert decision.model.logical_name == "coding"

    def test_priority_tie_breaks_by_logical_name(self) -> None:
        """Lexicographic tie-break is deterministic."""
        model_a = ModelDefinition(
            logical_name="aaa-model",
            provider="ollama",
            provider_model="a",
            capabilities=frozenset({Capability.GENERAL}),
            priority=10,
        )
        model_z = ModelDefinition(
            logical_name="zzz-model",
            provider="ollama",
            provider_model="z",
            capabilities=frozenset({Capability.GENERAL}),
            priority=10,
        )
        router = _router([model_z, model_a])  # registered in reverse
        decision = router.route(RoutingRequest(task_type=TaskType.CHAT))
        # Same priority, lexicographic wins
        assert decision.model.logical_name == "aaa-model"

    def test_coding_task_prefers_high_priority_coding(self) -> None:
        router = _router([_low_priority_coding_model(), _coding_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CODING))
        assert decision.model.logical_name == "coding"
        assert decision.model.priority == 20


# ---------------------------------------------------------------------------
# Routing: preferred model
# ---------------------------------------------------------------------------


class TestRoutingPreferredModel:
    def test_preferred_model_honoured_when_capable(self) -> None:
        router = _router([_coding_model(), _general_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CODING,
                preferred_model="general",  # general has GENERAl, not CODING
            )
        )
        # general lacks CODING capability → must not be honoured
        # Router should fall back to coding model
        assert decision.model.logical_name == "coding"

    def test_preferred_model_honoured_when_satisfies_caps(self) -> None:
        """Preferred model is honoured when it satisfies all requirements."""
        router = _router([_coding_model(), _general_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CHAT,
                preferred_model="general",  # has GENERAL
            )
        )
        assert decision.model.logical_name == "general"
        assert decision.preferred_honoured is True

    def test_preferred_model_rejected_when_lacks_capability(self) -> None:
        """Preferred model is silently rejected when it lacks a required capability."""
        router = _router([_general_model()])  # only general
        with pytest.raises(NoSuitableModelError):
            router.route(
                RoutingRequest(
                    task_type=TaskType.CODING,
                    preferred_model="general",  # lacks CODING
                )
            )

    def test_preferred_model_rejected_when_disabled(self) -> None:
        router = _router([_disabled_model(), _coding_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CHAT,
                preferred_model="disabled-legacy",
            )
        )
        # Falls back to coding model
        assert decision.model.logical_name == "coding"
        assert decision.preferred_honoured is False

    def test_preferred_model_rejected_when_unknown(self) -> None:
        router = _router([_general_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CHAT,
                preferred_model="does-not-exist",
            )
        )
        assert decision.model.logical_name == "general"
        assert decision.preferred_honoured is False

    def test_preferred_model_rejected_when_vision_but_vision_task(self) -> None:
        """Cannot prefer a non-vision model for a vision task."""
        router = _router([_general_model(), _vision_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.VISION,
                preferred_model="general",  # no VISION capability
            )
        )
        assert decision.model.logical_name == "vision"
        assert decision.preferred_honoured is False


# ---------------------------------------------------------------------------
# Routing: excluded models
# ---------------------------------------------------------------------------


class TestRoutingExcludedModels:
    def test_excluded_model_not_selected(self) -> None:
        router = _router([_coding_model(), _general_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CHAT,
                excluded_models=frozenset({"coding"}),
            )
        )
        assert decision.model.logical_name == "general"

    def test_excluded_preferred_model_falls_back(self) -> None:
        router = _router([_coding_model(), _general_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CHAT,
                preferred_model="coding",
                excluded_models=frozenset({"coding"}),
            )
        )
        assert decision.model.logical_name == "general"
        assert decision.preferred_honoured is False


# ---------------------------------------------------------------------------
# Routing: modality enforcement
# ---------------------------------------------------------------------------


class TestRoutingModalities:
    def test_vision_modality_required(self) -> None:
        """Image modality must be satisfied."""
        router = _router([_general_model(), _vision_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.VISION,
                input_modalities=frozenset({Modality.IMAGE, Modality.TEXT}),
            )
        )
        assert decision.model.logical_name == "vision"

    def test_text_only_model_rejected_for_vision_task(self) -> None:
        """A text-only model cannot satisfy a vision modality requirement."""
        router = _router([_general_model()])  # text only
        with pytest.raises(NoSuitableModelError):
            router.route(
                RoutingRequest(
                    input_modalities=frozenset({Modality.IMAGE}),
                )
            )

    def test_modality_satisfied_field_reflects_match(self) -> None:
        router = _router([_vision_model()])
        decision = router.route(
            RoutingRequest(input_modalities=frozenset({Modality.IMAGE}))
        )
        assert decision.modality_satisfied is True


# ---------------------------------------------------------------------------
# Routing: explicit capabilities
# ---------------------------------------------------------------------------


class TestRoutingExplicitCapabilities:
    def test_explicit_coding_capability(self) -> None:
        router = _router([_coding_model(), _general_model()])
        decision = router.route(
            RoutingRequest(required_capabilities=frozenset({Capability.CODING}))
        )
        assert decision.model.logical_name == "coding"

    def test_explicit_general_capability(self) -> None:
        router = _router([_coding_model(), _general_model()])
        decision = router.route(
            RoutingRequest(required_capabilities=frozenset({Capability.GENERAL}))
        )
        # Both have GENERAl; priority wins
        assert decision.model.logical_name == "coding"
        assert Capability.GENERAL in decision.matched_capabilities

    def test_multiple_required_capabilities(self) -> None:
        """Request for multiple capabilities: both must be satisfied."""
        coding_with_vision = ModelDefinition(
            logical_name="coding-vision",
            provider="ollama",
            provider_model="llava-coder",
            capabilities=frozenset({Capability.CODING, Capability.VISION}),
            input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        router = _router([coding_with_vision, _coding_model(), _vision_model()])
        decision = router.route(
            RoutingRequest(
                required_capabilities=frozenset(
                    {Capability.CODING, Capability.VISION}
                ),
                input_modalities=frozenset({Modality.IMAGE}),
            )
        )
        assert decision.model.logical_name == "coding-vision"


# ---------------------------------------------------------------------------
# Routing: determinism
# ---------------------------------------------------------------------------


class TestRoutingDeterminism:
    def test_identical_inputs_produce_identical_outputs(self) -> None:
        router = _router([_coding_model(), _general_model()])
        req = RoutingRequest(task_type=TaskType.CHAT)
        d1 = router.route(req)
        d2 = router.route(req)
        assert d1.model.logical_name == d2.model.logical_name
        assert d1.score == d2.score
        assert d1.reason == d2.reason

    def test_deterministic_across_multiple_runs(self) -> None:
        models = [_coding_model(), _general_model()]
        req = RoutingRequest(task_type=TaskType.CHAT)
        for _ in range(10):
            router = _router(models.copy())
            decision = router.route(req)
            assert decision.model.logical_name == "coding"


# ---------------------------------------------------------------------------
# RoutingDecision metadata
# ---------------------------------------------------------------------------


class TestRoutingDecisionMetadata:
    def test_decision_has_reason(self) -> None:
        router = _router([_coding_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CODING))
        assert decision.reason
        assert len(decision.reason) > 10
        assert "coding" in decision.reason

    def test_decision_has_score(self) -> None:
        router = _router([_coding_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CODING))
        assert isinstance(decision.score, int)
        assert decision.score > 0

    def test_decision_has_matched_capabilities(self) -> None:
        router = _router([_coding_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CODING))
        assert Capability.CODING in decision.matched_capabilities

    def test_decision_modality_satisfied(self) -> None:
        router = _router([_vision_model()])
        decision = router.route(
            RoutingRequest(
                input_modalities=frozenset({Modality.IMAGE}),
            )
        )
        assert decision.modality_satisfied is True

    def test_preferred_honoured_flag(self) -> None:
        router = _router([_general_model()])
        decision = router.route(
            RoutingRequest(
                task_type=TaskType.CHAT,
                preferred_model="general",
            )
        )
        assert decision.preferred_honoured is True


# ---------------------------------------------------------------------------
# Score composition
# ---------------------------------------------------------------------------


class TestRoutingScore:
    def test_priority_contributes_to_score(self) -> None:
        router = _router([_low_priority_coding_model(), _coding_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CODING))
        assert decision.model.logical_name == "coding"
        # coding has priority 20, coding-fallback has priority 5
        assert decision.score >= 100  # base score + priority

    def test_disabled_model_not_in_score_computation(self) -> None:
        """Disabled model should not affect routing outcome."""
        router = _router([_disabled_model(), _general_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CHAT))
        assert decision.model.logical_name == "general"


# ---------------------------------------------------------------------------
# RoutingRequest defaults
# ---------------------------------------------------------------------------


class TestRoutingRequestDefaults:
    def test_default_task_type_is_chat(self) -> None:
        req = RoutingRequest()
        assert req.task_type == TaskType.CHAT

    def test_default_capabilities_derived_from_task_type(self) -> None:
        router = _router([_general_model()])
        decision = router.route(RoutingRequest(task_type=TaskType.CHAT))
        assert decision.model.logical_name == "general"

    def test_explicit_empty_capabilities(self) -> None:
        """Explicit empty set means no capability requirement."""
        router = _router([_general_model()])
        # Should work with no required caps
        decision = router.route(
            RoutingRequest(
                required_capabilities=frozenset(),
            )
        )
        assert decision.model.logical_name == "general"


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class TestRoutingErrors:
    def test_no_suitable_model_error_is_routing_error(self) -> None:
        router = _router([_general_model()])
        with pytest.raises(RoutingError):
            router.route(RoutingRequest(task_type=TaskType.VISION))

    def test_no_suitable_model_error_from_core_routing_errors(self) -> None:
        router = _router([_general_model()])
        with pytest.raises(NoSuitableModelError):
            router.route(RoutingRequest(task_type=TaskType.VISION))

    def test_no_suitable_model_error_from_llm_re_exports(self) -> None:
        # Verify the re-export from core.llm works
        from core.llm import NoSuitableModelError as LLMNoSuitableModelError

        router = _router([_general_model()])
        with pytest.raises(LLMNoSuitableModelError):
            router.route(RoutingRequest(task_type=TaskType.VISION))

    def test_invalid_request_type_raises(self) -> None:
        router = _router([_general_model()])
        with pytest.raises(TypeError, match="RoutingRequest"):
            router.route({"task_type": "chat"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_load_default_registry_returns_non_empty_registry(self) -> None:
        reg = load_default_registry()
        assert len(reg) >= 3
        assert reg.has("coding")
        assert reg.has("general")
        assert reg.has("vision")

    def test_default_registry_has_enabled_models(self) -> None:
        reg = load_default_registry()
        enabled = reg.enabled()
        assert len(enabled) >= 3
        assert all(m.enabled for m in enabled)

    def test_load_default_registry_falls_back_to_builtin_when_file_missing(self) -> None:
        """load_default_registry returns the built-in registry when the path doesn't exist."""
        # We test this by passing a path that definitely doesn't exist
        # (and can't be created, e.g. a system directory on restricted environments).
        # The function checks file_path.is_file() and falls back to builtin.
        reg = load_default_registry("Z:/nonexistent/path/that/is/not/a/file.yaml")
        assert len(reg) >= 3
        assert reg.has("coding")


# ---------------------------------------------------------------------------
# Logging (security: no sensitive content)
# ---------------------------------------------------------------------------


class TestRoutingLogsNoSensitiveContent:
    def test_route_logs_task_type_not_content(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO)
        router = _router([_coding_model()])
        router.route(RoutingRequest(task_type=TaskType.CODING))
        log_messages = [r.message for r in caplog.records]
        assert any("coding" in msg for msg in log_messages)
        # No task content should be in logs
        for msg in log_messages:
            assert "secret" not in msg.lower()
            assert "password" not in msg.lower()
