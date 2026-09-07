"""Exception hierarchy for the model layer.

The application layer should catch `LLMError` and re-raise as a domain-specific
error or convert to an HTTP response. Provider-specific exceptions (httpx,
requests, SDK errors) should never propagate past the provider boundary.
"""
from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """Base class for all model-layer errors."""


class ConfigurationError(LLMError):
    """The gateway or provider is misconfigured.

    Examples: unknown provider name, missing required configuration,
    a base_url pointing at a non-loopback host.
    """


class ProviderUnavailableError(LLMError):
    """The provider endpoint is unreachable (connection refused, DNS failure)."""


class ProviderTimeoutError(LLMError):
    """A request to the provider timed out."""


class ModelNotFoundError(LLMError):
    """The requested model is not available on the provider."""

    def __init__(self, model: str, message: Optional[str] = None) -> None:
        self.model = model
        if message is None:
            message = f"Model '{model}' not found on provider"
        super().__init__(message)


class ProviderResponseError(LLMError):
    """The provider returned a response we could not parse or that was malformed."""


class ProviderError(LLMError):
    """A catch-all for unexpected upstream provider failures.

    Use this when the provider returned a non-2xx status (other than 404 for a
    missing model), or when an unexpected exception occurred in the provider
    adapter.
    """
