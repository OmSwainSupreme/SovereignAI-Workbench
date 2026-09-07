"""Configuration-driven model registry.

The registry is the single source of truth for the set of models the router
may consider. New models are added by **configuration**, not by code changes.

There are two ways to populate a registry:

1. **Programmatic** (used by tests and callers that build the registry
   in-memory)::

       from core.routing import Capability, Modality, ModelDefinition
       from core.routing.registry import ModelRegistry

       registry = ModelRegistry()
       registry.register(ModelDefinition(
           logical_name="coding",
           provider="ollama",
           provider_model="qwen2.5-coder:3b",
           capabilities=frozenset({Capability.CODING, Capability.GENERAL}),
       ))

2. **Configuration-driven** (the default for the application)::

       from core.routing.registry import load_default_registry
       registry = load_default_registry("config/models.yaml")

The router never talks to a provider directly; it only reads from a
:class:`ModelRegistry` instance. This means the router is fully testable
without any provider backend being reachable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Union

from core.routing.errors import RoutingConfigurationError
from core.routing.types import (
    Capability,
    Modality,
    ModelDefinition,
)


# A path or path-like (string or os.PathLike) is acceptable.
PathLike = Union[str, Path]


class ModelRegistry:
    """A configuration-driven collection of :class:`ModelDefinition` entries.

    The registry enforces:

    * Each ``logical_name`` is unique (re-registering replaces the prior entry
      and is permitted, mirroring :class:`core.llm.ProviderRegistry`).
    * The registry is non-empty before it can be used for routing — an empty
      registry is a configuration error, not a silent "no models" state.

    Instances are safe for concurrent read access once constructed. They are
    not designed for concurrent mutation; build the registry at startup and
    reuse it.
    """

    def __init__(self, models: Iterable[ModelDefinition] = ()) -> None:
        self._models: dict[str, ModelDefinition] = {}
        for model in models:
            self.register(model)

    # ------------------------------------------------------------------ Mutate

    def register(self, model: ModelDefinition) -> None:
        """Add or replace a model entry by its ``logical_name``."""
        if not model.logical_name:
            raise RoutingConfigurationError(
                "ModelDefinition.logical_name must be a non-empty string"
            )
        if not model.provider:
            raise RoutingConfigurationError(
                f"ModelDefinition for {model.logical_name!r} has no provider"
            )
        if not model.provider_model:
            raise RoutingConfigurationError(
                f"ModelDefinition for {model.logical_name!r} has no provider_model"
            )
        self._models[model.logical_name] = model

    def unregister(self, logical_name: str) -> None:
        """Remove a model by its logical name. No-op if absent."""
        self._models.pop(logical_name, None)

    # -------------------------------------------------------------------- Read

    def has(self, logical_name: str) -> bool:
        return logical_name in self._models

    def get(self, logical_name: str) -> ModelDefinition:
        try:
            return self._models[logical_name]
        except KeyError as exc:
            raise RoutingConfigurationError(
                f"Unknown logical model: '{logical_name}'. "
                f"Registered models: {sorted(self.names()) or '<none>'}"
            ) from exc

    def names(self) -> list[str]:
        """Return the registered logical names in insertion order."""
        return list(self._models.keys())

    def all(self) -> list[ModelDefinition]:
        """Return all registered models as a list (in insertion order)."""
        return list(self._models.values())

    def enabled(self) -> list[ModelDefinition]:
        """Return only enabled models (in insertion order)."""
        return [m for m in self._models.values() if m.enabled]

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, logical_name: object) -> bool:
        return isinstance(logical_name, str) and logical_name in self._models


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _parse_capability(value: str) -> Capability:
    try:
        return Capability(value)
    except ValueError as exc:
        valid = ", ".join(c.value for c in Capability)
        raise RoutingConfigurationError(
            f"Unknown capability {value!r}. Valid: {valid}"
        ) from exc


def _parse_modality(value: str) -> Modality:
    try:
        return Modality(value)
    except ValueError as exc:
        valid = ", ".join(m.value for m in Modality)
        raise RoutingConfigurationError(
            f"Unknown modality {value!r}. Valid: {valid}"
        ) from exc


def _model_from_dict(entry: Mapping[str, object]) -> ModelDefinition:
    """Construct a :class:`ModelDefinition` from a config-file dict."""
    required_fields = ("logical_name", "provider", "provider_model")
    missing = [f for f in required_fields if f not in entry]
    if missing:
        raise RoutingConfigurationError(
            f"Model entry is missing required field(s): {', '.join(missing)}. "
            f"Entry: {dict(entry)}"
        )

    raw_caps = entry.get("capabilities", [])
    if not isinstance(raw_caps, list):
        raise RoutingConfigurationError(
            f"Model {entry['logical_name']!r}: 'capabilities' must be a list, "
            f"got {type(raw_caps).__name__}"
        )
    capabilities = frozenset(_parse_capability(str(c)) for c in raw_caps)

    raw_input = entry.get("input_modalities", [Modality.TEXT.value])
    if not isinstance(raw_input, list):
        raise RoutingConfigurationError(
            f"Model {entry['logical_name']!r}: 'input_modalities' must be a list"
        )
    input_modalities = frozenset(_parse_modality(str(m)) for m in raw_input)

    raw_output = entry.get("output_modalities", [Modality.TEXT.value])
    if not isinstance(raw_output, list):
        raise RoutingConfigurationError(
            f"Model {entry['logical_name']!r}: 'output_modalities' must be a list"
        )
    output_modalities = frozenset(_parse_modality(str(m)) for m in raw_output)

    priority_raw = entry.get("priority", 0)
    try:
        priority = int(priority_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RoutingConfigurationError(
            f"Model {entry['logical_name']!r}: 'priority' must be an integer"
        ) from exc

    context_raw = entry.get("context_length", None)
    context_length: int | None
    if context_raw is None or context_raw == "":
        context_length = None
    else:
        try:
            context_length = int(context_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RoutingConfigurationError(
                f"Model {entry['logical_name']!r}: 'context_length' must be an integer"
            ) from exc

    return ModelDefinition(
        logical_name=str(entry["logical_name"]),
        provider=str(entry["provider"]),
        provider_model=str(entry["provider_model"]),
        capabilities=capabilities,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        enabled=bool(entry.get("enabled", True)),
        priority=priority,
        context_length=context_length,
        description=str(entry.get("description", "")),
    )


def load_registry_from_mapping(data: Mapping[str, object]) -> ModelRegistry:
    """Load a registry from an already-parsed config mapping.

    Expected shape::

        models:
          - logical_name: coding
            provider: ollama
            provider_model: qwen2.5-coder:7b
            capabilities: [coding, general]
            input_modalities: [text]
            enabled: true
            priority: 10
            context_length: 32768
            description: Qwen 2.5 Coder 7B
    """
    if "models" not in data:
        raise RoutingConfigurationError(
            "Routing config must contain a top-level 'models' key"
        )
    raw_models = data["models"]
    if not isinstance(raw_models, list):
        raise RoutingConfigurationError(
            f"'models' must be a list, got {type(raw_models).__name__}"
        )

    registry = ModelRegistry()
    for entry in raw_models:
        if not isinstance(entry, Mapping):
            raise RoutingConfigurationError(
                f"Each model entry must be a mapping, got {type(entry).__name__}"
            )
        registry.register(_model_from_dict(entry))
    return registry


def load_registry_from_path(path: PathLike) -> ModelRegistry:
    """Load a registry from a YAML or JSON file on disk.

    The file format is selected by extension: ``.yaml`` / ``.yml`` use YAML,
    anything else falls back to JSON. YAML support requires PyYAML to be
    installed; if the import fails for a YAML file, the user sees a clear
    error rather than a silent fallback to JSON.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RoutingConfigurationError(
                f"YAML routing config requested but PyYAML is not installed: {exc}"
            ) from exc
        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise RoutingConfigurationError(
            f"Routing config root must be a mapping, got {type(data).__name__}"
        )
    return load_registry_from_mapping(data)


# ---------------------------------------------------------------------------
# Built-in default registry (config-free)
# ---------------------------------------------------------------------------


def _builtin_default_registry() -> ModelRegistry:
    """A reasonable default registry, used when no config file is provided.

    This is what the application uses by default. It is **not** a model
    download manifest — these are just routing entries; whether the actual
    model weights are present on the provider is the provider's problem.
    """
    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
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
            context_length=32768,
            description="General-purpose chat and reasoning model.",
        )
    )
    registry.register(
        ModelDefinition(
            logical_name="coding",
            provider="ollama",
            provider_model="qwen2.5-coder:3b",
            capabilities=frozenset({Capability.CODING, Capability.GENERAL}),
            input_modalities=frozenset({Modality.TEXT}),
            output_modalities=frozenset({Modality.TEXT}),
            enabled=True,
            priority=20,
            context_length=32768,
            description="Code generation and refactoring model.",
        )
    )
    registry.register(
        ModelDefinition(
            logical_name="vision",
            provider="ollama",
            provider_model="qwen2.5vl:3b",
            capabilities=frozenset({Capability.VISION, Capability.GENERAL}),
            input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
            output_modalities=frozenset({Modality.TEXT}),
            enabled=True,
            priority=20,
            context_length=32768,
            description="Image and visual understanding model.",
        )
    )
    return registry


def load_default_registry(path: PathLike | None = None) -> ModelRegistry:
    """Load the application's default model registry.

    If ``path`` is given and the file exists, it is used. Otherwise the
    built-in default registry is returned. This keeps the router usable
    out of the box while still allowing production overrides via config.
    """
    if path is not None:
        file_path = Path(path)
        if file_path.is_file():
            return load_registry_from_path(file_path)
    return _builtin_default_registry()
