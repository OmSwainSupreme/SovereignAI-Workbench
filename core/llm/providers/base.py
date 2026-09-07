"""Abstract base class for model providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from core.llm.types import (
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderHealth,
    StreamChunk,
)


class BaseProvider(ABC):
    """Provider-agnostic interface to a model backend.

    Implementations encapsulate everything specific to a particular model
    serving system (Ollama, llama.cpp, vLLM, …). The ModelGateway never
    imports a concrete implementation directly.

    Implementations MUST:

    * translate provider-specific exceptions into the ``LLMError`` hierarchy;
    * refuse to operate against a non-loopback endpoint when the protocol
      requires a local server (this is enforced by ``OllamaProvider`` and any
      future local-only provider);
    * never log prompts, completions, or any other potentially sensitive
      content;
    * be safe for concurrent use.
    """

    #: A short, human-readable name for this provider class. Used in logs.
    name: str = "base"

    #: The default model to use when a ``GenerationRequest`` does not specify one.
    default_model: Optional[str] = None

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single response to a chat request."""

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Stream tokens from the model.

        Default implementation raises ``NotImplementedError``; subclasses
        that support streaming should override.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support streaming generation"
        )
        # Make this an async generator for type-checkers.
        if False:  # pragma: no cover
            yield ""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Return the health status of the provider backend."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return the list of models available on the provider."""

    async def close(self) -> None:
        """Release any resources held by the provider.

        Call this at application shutdown. The default implementation is a no-op.
        """

