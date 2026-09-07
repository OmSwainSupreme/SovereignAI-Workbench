"""SovereignAI Workbench — model routing layer.

Public surface for the capability-based Model Router. The router is a
**selector** — it does not call any provider itself. The selection is passed
to a :class:`core.llm.ModelGateway` for actual generation.

The router is deterministic: identical input always produces the same output,
given the same configured model registry. It never calls an LLM to decide
which model to use.

Example::

    from core.routing import (
        ModelRouter,
        RoutingRequest,
        TaskType,
        Capability,
    )
    from core.routing.registry import ModelRegistry, load_default_registry

    registry = load_default_registry()
    router = ModelRouter(registry=registry)

    decision = router.route(
        RoutingRequest(
            task_type=TaskType.CODING,
            required_capabilities={Capability.CODING},
        )
    )
    print(decision.model.logical_name, decision.model.provider_model)
"""
from core.routing.errors import (
    NoSuitableModelError,
    RoutingConfigurationError,
    RoutingError,
)
from core.routing.router import ModelRouter
from core.routing.types import (
    Capability,
    Modality,
    ModelDefinition,
    RoutingDecision,
    RoutingRequest,
    TaskType,
)

__all__ = [
    # types
    "Capability",
    "Modality",
    "ModelDefinition",
    "RoutingDecision",
    "RoutingRequest",
    "TaskType",
    # router
    "ModelRouter",
    # errors
    "NoSuitableModelError",
    "RoutingConfigurationError",
    "RoutingError",
]
