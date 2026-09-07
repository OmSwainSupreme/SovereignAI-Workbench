"""Application settings loaded from environment variables.

The :class:`Settings` class is organised into flat groups for env-var
binding clarity:

* ``app_*``, ``host``, ``port``, ``log_level`` - application identity and runtime
* ``llm_provider`` - which provider to use (``"ollama"`` for this phase)
* ``llm_ollama_base_url`` / ``llm_ollama_default_model`` /
  ``llm_ollama_request_timeout_seconds`` - Ollama-specific configuration

Environment variable names use the ``__`` separator (pydantic-settings
convention), so the corresponding env vars are:

    LLM__PROVIDER
    LLM__OLLAMA__BASE_URL
    LLM__OLLAMA__DEFAULT_MODEL
    LLM__OLLAMA__REQUEST_TIMEOUT_SECONDS
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Registered provider names at the time settings are loaded.
# The provider registry is imported lazily to avoid circular deps.
_VALID_PROVIDER_NAMES: frozenset[str] = frozenset()


def _load_valid_provider_names() -> frozenset[str]:
    """Return the set of currently registered provider names."""
    try:
        # Avoid a circular import at module level.
        from core.llm.registry import default_registry

        return frozenset(default_registry.names())
    except Exception:
        # If the registry cant be loaded (e.g. during unit tests that
        # mock it), return an empty set and skip validation.
        return frozenset()


@dataclass(frozen=True)
class OllamaSettings:
    """Structured view of Ollama provider configuration (runtime access only)."""

    base_url: str
    default_model: str
    request_timeout_seconds: int


@dataclass(frozen=True)
class LLMSettings:
    """Structured view of the model gateway configuration (runtime access only)."""

    provider: str
    ollama: OllamaSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Application identity --------------------------------------------

    app_name: str = Field(
        default="SovereignAI Workbench",
        description="Display name for the application.",
    )
    app_version: str = Field(
        default="0.2.0",
        description="Current application version.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, etc.).",
    )

    # --- Server ----------------------------------------------------------

    host: str = Field(
        default="127.0.0.1",
        description="Server bind host.",
    )
    port: int = Field(
        default=8000,
        description="Server bind port.",
    )

    # --- Logging ---------------------------------------------------------

    log_level: str = Field(
        default="INFO",
        description="Application log level.",
    )

    # --- Model gateway (flat env-bound fields) ---------------------------

    llm_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("llm_provider", "LLM__PROVIDER"),
        description=(
            "Name of the LLM provider to use. Must be a registered provider "
            "(run ``core.llm.registry.default_registry.names()`` to list them). "
            "Invalid names cause a ConfigurationError at startup."
        ),
    )

    llm_ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("llm_ollama_base_url", "LLM__OLLAMA__BASE_URL"),
        description="Base URL of the local Ollama server (must be loopback).",
    )

    llm_ollama_default_model: str = Field(
        default="",
        validation_alias=AliasChoices(
            "llm_ollama_default_model", "LLM__OLLAMA__DEFAULT_MODEL"
        ),
        description=(
            "Default model to use when a request does not specify one. "
            "Example: 'llama3', 'mistral', 'phi3'. Leave empty to require the "
            "caller to specify a model."
        ),
    )

    llm_ollama_request_timeout_seconds: int = Field(
        default=120,
        ge=1,
        validation_alias=AliasChoices(
            "llm_ollama_request_timeout_seconds",
            "LLM__OLLAMA__REQUEST_TIMEOUT_SECONDS",
        ),
        description="HTTP request timeout in seconds for Ollama calls.",
    )

    # --- Configuration ---------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- Convenience accessors -------------------------------------------

    @property
    def llm(self) -> LLMSettings:
        """Return the structured LLM configuration."""
        return LLMSettings(
            provider=self.llm_provider,
            ollama=OllamaSettings(
                base_url=self.llm_ollama_base_url,
                default_model=self.llm_ollama_default_model,
                request_timeout_seconds=self.llm_ollama_request_timeout_seconds,
            ),
        )

    @field_validator("llm_provider", mode="after")
    @classmethod
    def _validate_provider_name(cls, value: str) -> str:
        """Fail fast if the configured provider is not registered."""
        global _VALID_PROVIDER_NAMES
        if not _VALID_PROVIDER_NAMES:
            _VALID_PROVIDER_NAMES = _load_valid_provider_names()
        if _VALID_PROVIDER_NAMES and value not in _VALID_PROVIDER_NAMES:
            from core.llm.errors import ConfigurationError

            raise ConfigurationError(
                f"LLM__PROVIDER={value!r} is not registered. "
                f"Registered providers: {sorted(_VALID_PROVIDER_NAMES)}. "
                f"Check for typos or ensure the provider module is importable."
            )
        return value


# Global singleton - imported wherever settings are needed.
settings = Settings()