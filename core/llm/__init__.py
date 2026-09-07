"""SovereignAI Workbench — model layer.

Public surface for the provider-agnostic model gateway.

Future code (agents, tools, etc.) should import from here, never from
``core.llm.providers.*`` or any concrete provider module.

Example::

    from core.llm import ModelGateway, GenerationRequest, ChatMessage, Role

    gateway = ModelGateway(provider_name="ollama", provider_config={"base_url": "http://127.0.0.1:11434", "default_model": "llama3"})
    response = await gateway.generate(GenerationRequest(messages=[ChatMessage.user("Hello!")]))
    print(response.content)
"""

# Re-export the routing layer from the sibling ``core.routing`` package so that
# callers who import from ``core.llm`` can also reach the router without knowing
# the exact sub-package path. The routing layer is deliberately in its own
# package (``core/routing/``) to keep the two concerns (gateway vs router)
# separate and independently testable.
from core.routing import (
    # types
    Capability,
    Modality,
    ModelDefinition,
    RoutingDecision,
    RoutingRequest,
    TaskType,
    # errors
    NoSuitableModelError,
    RoutingConfigurationError,
    RoutingError,
)
from core.llm.errors import (
    ConfigurationError,
    LLMError,
    ModelNotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from core.llm.gateway import ModelGateway
from core.llm.registry import default_registry, ProviderRegistry
from core.llm.types import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderHealth,
    Role,
    StreamChunk,
)

__all__ = [
    # types
    "ChatMessage",
    "GenerationRequest",
    "GenerationResponse",
    "ModelInfo",
    "ProviderHealth",
    "Role",
    "StreamChunk",
    # routing types
    "Capability",
    "Modality",
    "ModelDefinition",
    "RoutingDecision",
    "RoutingRequest",
    "TaskType",
    # routing errors
    "NoSuitableModelError",
    "RoutingConfigurationError",
    "RoutingError",
    # gateway
    "ModelGateway",
    # registry
    "ProviderRegistry",
    "default_registry",
    # errors
    "LLMError",
    "ConfigurationError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "ModelNotFoundError",
    "ProviderResponseError",
    "ProviderError",
]
