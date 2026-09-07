"""Tests for Ollama provider configuration.

These tests cover the OllamaConfig dataclass and the LLM/Ollama settings
loaded from environment variables. They do NOT require a running Ollama
server.
"""
from __future__ import annotations

import pytest

from backend.app.core.config import LLMSettings, OllamaSettings, Settings
from core.llm import ConfigurationError
from core.llm.providers.ollama import OllamaConfig, _validate_loopback


class TestValidateLoopback:
    def test_loopback_ipv4(self) -> None:
        assert _validate_loopback("http://127.0.0.1:11434") == "http://127.0.0.1:11434"

    def test_loopback_localhost(self) -> None:
        assert _validate_loopback("http://localhost:11434") == "http://localhost:11434"

    def test_loopback_ipv6(self) -> None:
        assert _validate_loopback("http://[::1]:11434") == "http://[::1]:11434"

    def test_loopback_with_trailing_slash(self) -> None:
        assert _validate_loopback("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"

    @pytest.mark.parametrize(
        "url",
        [
            "http://192.168.1.10:11434",
            "http://10.0.0.5:11434",
            "http://example.com:11434",
            "http://api.openai.com:443",
            "https://ollama.example.org",
        ],
    )
    def test_rejects_non_loopback(self, url: str) -> None:
        with pytest.raises(ConfigurationError, match="loopback"):
            _validate_loopback(url)

    def test_rejects_invalid_url(self) -> None:
        with pytest.raises(ConfigurationError):
            _validate_loopback("not a url")


class TestOllamaConfigDataclass:
    def test_defaults(self) -> None:
        cfg = OllamaConfig()
        assert cfg.base_url == "http://127.0.0.1:11434"
        assert cfg.default_model == ""
        assert cfg.request_timeout_seconds == 120

    def test_explicit_values(self) -> None:
        cfg = OllamaConfig(
            base_url="http://localhost:11434",
            default_model="llama3",
            request_timeout_seconds=300,
        )
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.default_model == "llama3"
        assert cfg.request_timeout_seconds == 300

    def test_minimum_timeout(self) -> None:
        # Values below 1 are clamped to 1.
        cfg = OllamaConfig(request_timeout_seconds=0)
        assert cfg.request_timeout_seconds == 1
        cfg2 = OllamaConfig(request_timeout_seconds=-5)
        assert cfg2.request_timeout_seconds == 1

    def test_rejects_non_loopback_url(self) -> None:
        with pytest.raises(ConfigurationError):
            OllamaConfig(base_url="http://192.168.1.1:11434")


class TestSettingsDefaults:
    def test_ollama_defaults_via_settings(self) -> None:
        s = Settings()
        assert s.llm_ollama_base_url == "http://127.0.0.1:11434"
        assert s.llm_ollama_default_model == ""
        assert s.llm_ollama_request_timeout_seconds == 120

    def test_llm_provider_default(self) -> None:
        s = Settings()
        assert s.llm_provider == "ollama"

    def test_structured_view(self) -> None:
        s = Settings()
        assert s.llm.provider == "ollama"
        assert s.llm.ollama.base_url == "http://127.0.0.1:11434"
        assert s.llm.ollama.default_model == ""
        assert s.llm.ollama.request_timeout_seconds == 120


class TestSettingsEnvLoading:
    def test_ollama_settings_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM__OLLAMA__BASE_URL", "http://127.0.0.1:9999")
        monkeypatch.setenv("LLM__OLLAMA__DEFAULT_MODEL", "mistral")
        monkeypatch.setenv("LLM__OLLAMA__REQUEST_TIMEOUT_SECONDS", "30")
        s = Settings()
        assert s.llm_ollama_base_url == "http://127.0.0.1:9999"
        assert s.llm_ollama_default_model == "mistral"
        assert s.llm_ollama_request_timeout_seconds == 30
        # Structured view reflects the same values
        assert s.llm.ollama.base_url == "http://127.0.0.1:9999"
        assert s.llm.ollama.default_model == "mistral"
        assert s.llm.ollama.request_timeout_seconds == 30

    def test_llm_provider_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM__PROVIDER", "ollama")
        s = Settings()
        assert s.llm_provider == "ollama"
        assert s.llm.provider == "ollama"

    def test_full_settings_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM__PROVIDER", "ollama")
        monkeypatch.setenv("LLM__OLLAMA__BASE_URL", "http://127.0.0.1:9999")
        monkeypatch.setenv("LLM__OLLAMA__DEFAULT_MODEL", "phi3")
        s = Settings()
        assert s.llm_provider == "ollama"
        assert s.llm.ollama.base_url == "http://127.0.0.1:9999"
        assert s.llm.ollama.default_model == "phi3"
# ---------------------------------------------------------------------------
# Provider name validation
# ---------------------------------------------------------------------------


class TestProviderNameValidation:
    def test_unknown_provider_name_raises_at_settings_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unknown LLM__PROVIDER value must fail immediately when Settings() is created."""
        monkeypatch.setenv("LLM__PROVIDER", "not-a-real-provider")
        with pytest.raises(ConfigurationError, match="not-a-real-provider"):
            Settings()

    def test_typo_provider_name_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A common typo (olama -> ollama) must be caught at settings load time."""
        monkeypatch.setenv("LLM__PROVIDER", "olama")
        with pytest.raises(ConfigurationError, match="olama"):
            Settings()

    def test_valid_ollama_provider_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The known-good value 'ollama' must still load successfully."""
        monkeypatch.setenv("LLM__PROVIDER", "ollama")
        s = Settings()
        assert s.llm_provider == "ollama"

    def test_error_message_lists_registered_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The error message must list available providers to help the operator."""
        monkeypatch.setenv("LLM__PROVIDER", "unknown-provider")
        with pytest.raises(ConfigurationError, match="ollama"):
            Settings()