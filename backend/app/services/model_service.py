"""Application-level service for accessing the model gateway and router.

This module is the **only** place in the backend that constructs or holds a
``ModelGateway`` or ``ModelRouter``. FastAPI routes and future business logic
obtain these through the ``get_model_service`` dependency, which is overridable
in tests via ``app.dependency_overrides``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from backend.app.core.config import OllamaSettings, Settings, settings
from core.llm import (
    ConfigurationError,
    LLMError,
    ModelGateway,
    ProviderHealth,
    RoutingRequest,
)
from core.llm.providers.ollama import OllamaConfig
from core.routing import ModelRouter
from core.routing.registry import load_default_registry


_logger = logging.getLogger("sovereign-ai.model_service")

#: In-process TTL for health/status results in seconds.
_HEALTH_TTL_SECONDS: float = 1.5


class _HealthCache:
    """A minimal in-process TTL cache for provider health status."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: ProviderHealth, ttl: float) -> None:
        self.value: ProviderHealth = value
        self.expires_at: float = time.monotonic() + ttl

    def is_stale(self) -> bool:
        return time.monotonic() > self.expires_at


class ModelService:
    """Application-facing wrapper around :class:`ModelGateway` and :class:`ModelRouter`.

    The router is available for future agent integration. The gateway is used
    for direct model access (Phase 2B behaviour).
    """

    def __init__(
        self,
        gateway: ModelGateway,
        router: Optional[ModelRouter] = None,
    ) -> None:
        self._gateway = gateway
        self._router = router
        self._health_cache: Optional[_HealthCache] = None

    @property
    def gateway(self) -> ModelGateway:
        """Return the underlying gateway (for advanced use cases / tests)."""
        return self._gateway

    @property
    def router(self) -> Optional[ModelRouter]:
        """Return the model router, if one was configured."""
        return self._router

    async def get_status(self) -> ProviderHealth:
        """Return the current provider health, cached for ~1-2 seconds.

        The cache prevents repeated status polling from hammering the local
        Ollama server when multiple callers (e.g. UI auto-refresh) query
        the endpoint simultaneously.
        """
        cached = self._health_cache
        if cached is not None and not cached.is_stale():
            _logger.debug("health.status  source=cache  provider=%s", cached.value.provider)
            return cached.value

        health = await self._gateway.health()
        self._health_cache = _HealthCache(health, _HEALTH_TTL_SECONDS)
        _logger.debug("health.status  source=fresh  provider=%s  reachable=%s", health.provider, health.reachable)
        return health


# ---------------------------------------------------------------------------
# Factory + dependency injection
# ---------------------------------------------------------------------------


def _build_provider_config(provider_name: str, app_settings: Settings) -> dict:
    """Translate the application settings into a provider-specific config dict."""
    if provider_name == "ollama":
        cfg: OllamaSettings = app_settings.llm.ollama
        return {
            "base_url": cfg.base_url,
            "default_model": cfg.default_model,
            "request_timeout_seconds": cfg.request_timeout_seconds,
        }
    raise ConfigurationError(
        f"Application does not know how to configure provider '{provider_name}'"
    )


def _build_model_router(app_settings: Optional[Settings] = None) -> Optional[ModelRouter]:
    """Build a :class:`ModelRouter` from the routing config file.

    Returns None if the routing config file does not exist. The router is
    optional — the service still works without it (for Phase 2B callers that
    bypass routing).
    """
    # Look for the routing config alongside the application root.
    routing_config = Path(__file__).parents[2] / "config" / "models.yaml"
    if not routing_config.is_file():
        _logger.debug(
            "Routing config not found at %s; ModelRouter not initialised",
            routing_config,
        )
        return None

    try:
        from core.routing.registry import load_registry_from_path
        registry = load_registry_from_path(routing_config)
        router = ModelRouter(registry=registry)
        _logger.info(
            "ModelRouter initialised with %d model(s): %s",
            len(registry),
            registry.names(),
        )
        return router
    except Exception as exc:
        _logger.warning(
            "Failed to load routing config %s: %s; ModelRouter not initialised",
            routing_config,
            exc,
        )
        return None


def build_model_service(app_settings: Optional[Settings] = None) -> ModelService:
    """Build a :class:`ModelService` from the application settings.

    The gateway is constructed lazily on first use, so creating a ModelService
    does not require the provider to be reachable. The router is also optional.
    """
    app_settings = app_settings or settings
    provider_name = app_settings.llm.provider
    provider_config = _build_provider_config(provider_name, app_settings)
    _logger.debug(
        "Building ModelService: provider=%s  config_keys=%s",
        provider_name,
        sorted(provider_config.keys()),
    )
    gateway = ModelGateway(
        provider_name=provider_name,
        provider_config=provider_config,
    )
    router = _build_model_router(app_settings)
    return ModelService(gateway=gateway, router=router)


_default_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """FastAPI dependency that returns the process-wide ModelService.

    The service is created on first call and cached. Tests can override
    this dependency to inject fakes.
    """
    global _default_service
    if _default_service is None:
        _default_service = build_model_service()
    return _default_service