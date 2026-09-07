"""Exception hierarchy for the routing layer.

These errors are part of the public surface of the routing layer. The
application layer should catch :class:`RoutingError` and translate to a
domain-specific error or HTTP response as appropriate.
"""
from __future__ import annotations

from typing import Optional


class RoutingError(Exception):
    """Base class for all routing errors."""


class RoutingConfigurationError(RoutingError):
    """The router or its registry is misconfigured.

    Examples: an empty registry, a model entry missing a logical name, an
    attempt to register two models with the same logical name.
    """


class NoSuitableModelError(RoutingError):
    """No configured model satisfies the routing request.

    Attributes:
        task_type: The task type from the failing request.
        required_capabilities: Capabilities the request required.
        input_modalities: Modalities the request required.
        reason: A short diagnostic explaining why no model matched.
            Safe to surface to end users.
    """

    def __init__(
        self,
        task_type: object,
        required_capabilities: object,
        input_modalities: object,
        reason: str,
        message: Optional[str] = None,
    ) -> None:
        self.task_type = task_type
        self.required_capabilities = required_capabilities
        self.input_modalities = input_modalities
        self.reason = reason
        if message is None:
            message = f"No suitable model for routing request: {reason}"
        super().__init__(message)
