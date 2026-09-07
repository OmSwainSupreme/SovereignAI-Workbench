"""Framework-agnostic data types for the model layer.

These types are deliberately plain dataclasses — they must not depend on FastAPI,
Pydantic, or any other framework. The application layer (backend/app) may wrap
them in Pydantic DTOs for HTTP transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    """Speaker role in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a chat-style conversation."""

    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Shorthand for a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Shorthand for a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Shorthand for an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)


@dataclass
class GenerationRequest:
    """A request to a model provider for a single response.

    The provider is free to interpret optional fields (temperature, top_p, etc.)
    as it sees fit. Providers that do not support a given option should ignore
    it rather than raising.
    """

    messages: list[ChatMessage]
    model: Optional[str] = None  # If None, provider uses its default
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None


@dataclass
class GenerationResponse:
    """A single (non-streaming) response from a model provider."""

    content: str
    model: str
    # Provider-specific opaque metadata (e.g. token counts, finish reason).
    # The application layer should not depend on its shape.
    raw: Optional[dict] = None
    finish_reason: Optional[str] = None


@dataclass
class StreamChunk:
    """A single chunk of a streaming response."""

    content: str
    model: str
    done: bool = False


@dataclass
class ModelInfo:
    """Describes a single model exposed by a provider."""

    name: str
    size_bytes: Optional[int] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None
    modified_at: Optional[str] = None


@dataclass
class ProviderHealth:
    """Provider health information returned by the gateway."""

    provider: str
    reachable: bool
    model_count: int = 0
    error: Optional[str] = None
    raw: Optional[dict] = field(default=None)
