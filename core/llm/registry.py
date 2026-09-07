"""Provider registry — maps a configured provider name to a provider factory.

A new provider (e.g. llama.cpp, vLLM) is added by:

    1. Creating ``core/llm/providers/<name>.py`` with a class implementing
       ``BaseProvider``.
    2. Calling ``default_registry.register("<name>", factory)`` in that module
       (or in a registration helper imported by the gateway).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from core.llm.errors import ConfigurationError
from core.llm.providers.base import BaseProvider


@dataclass(frozen=True)
class ProviderFactory:
    """A factory function plus a human-readable description for diagnostics."""

    name: str
    factory: Callable[..., BaseProvider]
    description: str = ""


class ProviderRegistry:
    """An in-process registry of provider factories keyed by name."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., BaseProvider],
        description: str = "",
    ) -> None:
        """Register a provider factory under the given name.

        Re-registering an existing name overwrites the prior entry. The registry
        never raises on overwrite — providers may be swapped in tests.
        """
        if not name:
            raise ValueError("Provider name must be a non-empty string")
        self._factories[name] = ProviderFactory(
            name=name, factory=factory, description=description
        )

    def unregister(self, name: str) -> None:
        """Remove a registered provider. No-op if absent."""
        self._factories.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._factories

    def names(self) -> list[str]:
        """Return the registered provider names in insertion order."""
        return list(self._factories.keys())

    def get(self, name: str) -> ProviderFactory:
        """Return the factory registered under ``name``.

        Raises:
            ConfigurationError: if no provider is registered under that name.
        """
        try:
            return self._factories[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown LLM provider: '{name}'. "
                f"Registered providers: {sorted(self.names()) or '<none>'}"
            ) from exc

    def create(self, name: str, **kwargs: object) -> BaseProvider:
        """Instantiate the provider registered under ``name``."""
        factory = self.get(name)
        return factory.factory(**kwargs)


# The application-level default registry. Providers are added to this registry
# when their module is imported. The gateway uses this registry unless a
# different one is injected (useful for tests).
default_registry = ProviderRegistry()


def _register_builtin_providers() -> None:
    """Register the providers that ship with SovereignAI Workbench."""
    # Local import avoids a circular dependency at module load.
    from core.llm.providers.ollama import OllamaProvider

    default_registry.register(
        "ollama",
        OllamaProvider,
        description=(
            "Local Ollama server (open-weight LLMs served by the Ollama daemon). "
            "Communicates only with a configured loopback HTTP endpoint."
        ),
    )


_register_builtin_providers()
