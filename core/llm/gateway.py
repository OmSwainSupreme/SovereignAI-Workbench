"""ModelGateway - the public, application-level interface for model access.

Future business logic (agents, tools, etc.) should import from this module, not
from ``core.llm.providers.*`` or any provider-specific module.

The gateway:

* selects the configured provider via a ``ProviderRegistry``;
* delegates generation and health calls to that provider;
* translates any ``LLMError`` exceptions upward without modification;
* logs structured metadata about each call (provider, model, latency, success).
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Optional

from core.llm.errors import LLMError
from core.llm.registry import ProviderRegistry, default_registry
from core.llm.types import (
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderHealth,
)
from core.llm.providers.base import BaseProvider


_logger = logging.getLogger("sovereign-ai.gateway")


class ModelGateway:
    """A gateway that delegates model requests to a configured provider.

    The gateway is constructed once (typically at application startup) and reused
    for the lifetime of the process. It is safe for concurrent use.

    Args:
        provider_name: Name of the provider to use (e.g. ``"ollama"``).
        provider_config: Provider-specific configuration dict.
            The ``create()`` method of the chosen provider interprets these.
        registry: The provider registry to use. Defaults to
            ``default_registry`` (which includes all built-in providers).
    """

    def __init__(
        self,
        provider_name: str,
        provider_config: dict,
        registry: Optional[ProviderRegistry] = None,
    ) -> None:
        self._provider_name = provider_name
        self._provider_config = provider_config
        self._registry = registry or default_registry
        self._provider: Optional[BaseProvider] = None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _get_provider(self) -> BaseProvider:
        """Lazily instantiate and return the backing provider."""
        if self._provider is None:
            _logger.debug(
                "Initialising provider %r with config: %s",
                self._provider_name,
                _config_summary(self._provider_config),
            )
            self._provider = self._registry.create(
                self._provider_name, **self._provider_config
            )
        return self._provider

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single response to a chat request (non-streaming)."""
        start = time.monotonic()
        provider = self._get_provider()
        model = request.model or provider.default_model or "(default)"
        success = False
        error_type: Optional[str] = None
        result: Optional[GenerationResponse] = None

        try:
            _logger.info(
                "generate.start  provider=%s  model=%s  n_messages=%d",
                self._provider_name,
                model,
                len(request.messages),
            )
            result = await provider.generate(request)
            success = True
            return result
        except Exception as exc:
            # Capture the exception type from the live exception object, not
            # via sys.exc_info() which is racy in async contexts.  Re-raise
            # immediately; the finally block logs metadata only.
            error_type = type(exc).__name__
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            if success and result is not None:
                _logger.info(
                    "generate.done  provider=%s  model=%s  latency_ms=%.1f  n_tokens_approx=%d",
                    self._provider_name,
                    result.model,
                    latency_ms,
                    len(result.content.split()),
                )
            else:
                _logger.warning(
                    "generate.fail  provider=%s  model=%s  latency_ms=%.1f  error=%s",
                    self._provider_name,
                    model,
                    latency_ms,
                    error_type or "unknown",
                )

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Stream tokens from the model.

        Not all providers support streaming; those that do not will raise
        ``NotImplementedError``. The caller should check capability before
        calling this method if needed.
        """
        start = time.monotonic()
        provider = self._get_provider()
        model = request.model or provider.default_model or "(default)"
        total_tokens = 0
        success = False
        error_type: Optional[str] = None

        try:
            _logger.info(
                "stream.start  provider=%s  model=%s  n_messages=%d",
                self._provider_name,
                model,
                len(request.messages),
            )
            async for token in provider.stream(request):
                total_tokens += 1
                success = True
                yield token
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            if not success:
                _logger.warning(
                    "stream.fail  provider=%s  model=%s  latency_ms=%.1f  error=%s",
                    self._provider_name,
                    model,
                    latency_ms,
                    error_type or "unknown",
                )
            else:
                _logger.info(
                    "stream.done  provider=%s  model=%s  latency_ms=%.1f  tokens=%d",
                    self._provider_name,
                    model,
                    latency_ms,
                    total_tokens,
                )

    async def health(self) -> ProviderHealth:
        """Return the health status of the configured provider.

        Raises:
            LLMError: on error (does not raise on provider being unhealthy;
                the returned ``ProviderHealth`` carries the error).
        """
        provider = self._get_provider()
        try:
            _logger.debug(
                "health.check  provider=%s",
                self._provider_name,
            )
            health = await provider.health()
            _logger.info(
                "health.done  provider=%s  reachable=%s  model_count=%d",
                self._provider_name,
                health.reachable,
                health.model_count,
            )
            return health
        except LLMError as exc:
            _logger.warning(
                "health.fail  provider=%s  error=%s",
                self._provider_name,
                type(exc).__name__,
            )
            raise

    async def list_models(self) -> list[ModelInfo]:
        """Return the list of models available from the provider."""
        provider = self._get_provider()
        try:
            _logger.debug(
                "list_models  provider=%s",
                self._provider_name,
            )
            models = await provider.list_models()
            _logger.info(
                "list_models.done  provider=%s  count=%d",
                self._provider_name,
                len(models),
            )
            return models
        except LLMError as exc:
            _logger.warning(
                "list_models.fail  provider=%s  error=%s",
                self._provider_name,
                type(exc).__name__,
            )
            raise


def _config_summary(config: dict) -> dict:
    """Return a sanitised copy of a config dict for logging (no secrets)."""
    REDACT = "<redacted>"
    return {
        k: (
            REDACT
            if any(t in k.lower() for t in ("key", "secret", "token", "password"))
            else v
        )
        for k, v in config.items()
    }