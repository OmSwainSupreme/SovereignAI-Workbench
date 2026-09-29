"""Tests for core.llm.types dataclasses."""
from __future__ import annotations

from core.llm.types import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderHealth,
    Role,
    StreamChunk,
)


class TestRole:
    def test_role_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"

    def test_role_from_string(self) -> None:
        assert Role("system") == Role.SYSTEM
        assert Role("user") == Role.USER
        assert Role("assistant") == Role.ASSISTANT


class TestChatMessage:
    def test_construct_message(self) -> None:
        msg = ChatMessage(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_shorthand_system(self) -> None:
        msg = ChatMessage.system("You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."

    def test_shorthand_user(self) -> None:
        msg = ChatMessage.user("What is 2+2?")
        assert msg.role == Role.USER
        assert msg.content == "What is 2+2?"

    def test_shorthand_assistant(self) -> None:
        msg = ChatMessage.assistant("The answer is 4.")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "The answer is 4."

    def test_message_is_frozen(self) -> None:
        msg = ChatMessage(role=Role.USER, content="Hello")
        with __import__("pytest").raises(AttributeError):
            msg.content = "Changed"


class TestGenerationRequest:
    def test_minimal_request(self) -> None:
        req = GenerationRequest(messages=[ChatMessage.user("Hello")])
        assert len(req.messages) == 1
        assert req.model is None
        assert req.temperature is None

    def test_full_request(self) -> None:
        req = GenerationRequest(
            messages=[ChatMessage.user("Hello")],
            model="llama3",
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["END"],
        )
        assert req.model == "llama3"
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.max_tokens == 256
        assert req.stop == ["END"]


class TestGenerationResponse:
    def test_response_fields(self) -> None:
        resp = GenerationResponse(
            content="42",
            model="llama3",
            raw={"total_duration": 1_000_000},
            finish_reason="stop",
        )
        assert resp.content == "42"
        assert resp.model == "llama3"
        assert resp.raw == {"total_duration": 1_000_000}
        assert resp.finish_reason == "stop"


class TestStreamChunk:
    def test_defaults(self) -> None:
        chunk = StreamChunk(content="Hello", model="llama3")
        assert chunk.content == "Hello"
        assert chunk.model == "llama3"
        assert chunk.done is False

    def test_done_chunk(self) -> None:
        chunk = StreamChunk(content="!", model="llama3", done=True)
        assert chunk.done is True


class TestModelInfo:
    def test_required_fields(self) -> None:
        info = ModelInfo(name="llama3")
        assert info.name == "llama3"
        assert info.size_bytes is None
        assert info.parameter_size is None

    def test_all_fields(self) -> None:
        info = ModelInfo(
            name="llama3:8b",
            size_bytes=5_000_000_000,
            family="llama",
            parameter_size="8B",
            quantization_level="Q4_0",
            modified_at="2024-01-01T00:00:00Z",
        )
        assert info.parameter_size == "8B"
        assert info.quantization_level == "Q4_0"


class TestProviderHealth:
    def test_healthy(self) -> None:
        h = ProviderHealth(provider="ollama", reachable=True, model_count=3)
        assert h.reachable is True
        assert h.model_count == 3
        assert h.error is None

    def test_unreachable_with_error(self) -> None:
        h = ProviderHealth(
            provider="ollama",
            reachable=False,
            model_count=0,
            error="Connection refused",
        )
        assert h.reachable is False
        assert h.error == "Connection refused"
