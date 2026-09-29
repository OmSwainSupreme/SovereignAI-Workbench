"""Ollama provider for SovereignAI Workbench.

Communicates with a locally running Ollama server over HTTP. Never sends data
to external hosts; the ``OllamaConfig`` refuses any non-loopback base URL at
construction time.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncIterator, Optional

import httpx

from core.llm.errors import (
    ConfigurationError,
    LLMError,
    ModelNotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from core.llm.providers.base import BaseProvider
from core.llm.types import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderHealth,
)

_logger = logging.getLogger("sovereign-ai.provider.ollama")

# Hosts considered loopback (used for URL validation).
_LOOPBACK_RE = re.compile(
    r"^(?:127\.\d+\.\d+\.\d+|localhost|::1|\[::1\])$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class OllamaConfig:
    """Configuration for an Ollama provider instance."""

    base_url: str
    default_model: str
    request_timeout_seconds: int

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "",
        request_timeout_seconds: int = 1800,
    ) -> None:
        self.base_url = _validate_loopback(base_url)
        self.default_model = default_model
        self.request_timeout_seconds = max(1, request_timeout_seconds)

    def __repr__(self) -> str:
        return (
            f"OllamaConfig(base_url={self.base_url!r}, "
            f"default_model={self.default_model!r}, "
            f"request_timeout_seconds={self.request_timeout_seconds})"
        )


def _validate_loopback(url: str) -> str:
    """Validate that ``url`` points at a loopback host. Raise ConfigurationError otherwise."""
    try:
        parsed = httpx.URL(url)
        host = parsed.host or ""
        if not _LOOPBACK_RE.match(host):
            raise ConfigurationError(
                f"Ollama base_url must use a loopback host (127.0.0.1, ::1, localhost). "
                f"Got: {url!r}  This is a security requirement to prevent accidental "
                f"data exfiltration to external hosts."
            )
        return url.rstrip("/")
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid Ollama base_url: {url!r}") from exc


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class StreamChunk(str):
    """A streaming text chunk that preserves thinking metadata without breaking str consumers."""

    content: str
    thinking: str
    is_thinking: bool

    def __new__(
        cls,
        content: str = "",
        thinking: str = "",
        is_thinking: bool = False,
    ) -> StreamChunk:
        text = thinking if is_thinking else content
        obj = super().__new__(cls, text)
        obj.content = content
        obj.thinking = thinking
        obj.is_thinking = is_thinking
        return obj


class OllamaProvider(BaseProvider):
    """Model provider backed by a local Ollama server.

    The server MUST be running on the same machine (loopback) and reachable at
    the configured ``base_url``. This provider communicates exclusively with
    ``/api/chat`` (non-streaming), ``/api/chat`` (streaming SSE), and
    ``/api/tags`` (model enumeration). It will refuse to start if the base URL
    is not loopback.

    The constructor accepts the same keyword arguments as :class:`OllamaConfig`
    so it can be instantiated via the :class:`ProviderRegistry`'s
    ``create(name, **kwargs)`` pattern.
    """

    name: str = "ollama"
    default_model: Optional[str] = None

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "",
        request_timeout_seconds: int = 1800,
    ) -> None:
        self._config = OllamaConfig(
            base_url=base_url,
            default_model=default_model,
            request_timeout_seconds=request_timeout_seconds,
        )
        # Use separate connect/read/write timeouts.
        # For streaming, httpx applies the `read` timeout between successive
        # chunks — NOT to the total response duration. On CPU-only hardware
        # qwen3 generates at ~3-5 tok/s and can spend 60-120s in the thinking
        # phase before the first content token. A flat 120s timeout would kill
        # the stream mid-generation. We use a short connect timeout (fast-fail
        # if Ollama is down) and a generous per-chunk read timeout.
        _read_timeout = float(max(self._config.request_timeout_seconds, 1800))
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=_read_timeout,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self.default_model = self._config.default_model or None
        self.last_metadata: dict[str, Any] = {}

    async def close(self) -> None:
        """Close the underlying HTTP client. Call this at application shutdown."""
        await self._client.aclose()

    def _url(self, path: str) -> str:
        return f"{self._config.base_url}{path}"

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """POST to ``path`` and return the parsed JSON body.

        Raises:
            ProviderUnavailableError: on connection failure.
            ProviderTimeoutError: on timeout.
            ProviderResponseError: on non-2xx or JSON decode error.
            ProviderError: on unexpected HTTP errors and other httpx failures.
        """
        _logger.debug("POST %s  payload_keys=%s", path, list(payload.keys()))
        try:
            response = await self._client.post(
                self._url(path),
                json=payload,
                timeout=timeout,
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self._config.base_url}: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ProviderTimeoutError(
                f"Ollama request to {path} timed out after "
                f"{self._config.request_timeout_seconds}s"
            ) from exc
        except httpx.ConnectTimeout as exc:
            raise ProviderUnavailableError(
                f"Connection to Ollama timed out: {exc}"
            ) from exc
        except httpx.PoolTimeout as exc:
            # PoolTimeout is a subclass of TimeoutException but means the
            # connection pool is exhausted, not a response timeout.
            raise ProviderError(
                f"Ollama request failed (pool exhausted): {exc}"
            ) from exc
        except httpx.WriteTimeout as exc:
            # WriteTimeout is also a subclass of TimeoutException.
            raise ProviderError(
                f"Ollama request failed (write timeout): {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            # Catch the rest of the httpx exception hierarchy not covered above
            # (e.g. RemoteProtocolError, PoolTimeout, NetworkError).  These
            # indicate a real problem talking to the local server.
            raise ProviderError(
                f"Ollama request failed: {exc}"
            ) from exc

        if response.status_code == 404:
            model = payload.get("model", "(unknown)")
            raise ModelNotFoundError(model)
        if response.status_code >= 500:
            raise ProviderError(
                f"Ollama server error {response.status_code}: {response.text[:200]}"
            )
        if not response.is_success:
            raise ProviderResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"Ollama returned non-JSON response: {exc}. Body: {response.text[:200]}"
            ) from exc

    async def _get(
        self,
        path: str,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """GET from ``path`` and return the parsed JSON body."""
        _logger.debug("GET %s", path)
        try:
            response = await self._client.get(
                self._url(path),
                timeout=timeout,
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self._config.base_url}: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ProviderTimeoutError(
                f"Ollama request to {path} timed out after "
                f"{self._config.request_timeout_seconds}s"
            ) from exc
        except httpx.ConnectTimeout as exc:
            raise ProviderUnavailableError(
                f"Connection to Ollama timed out: {exc}"
            ) from exc
        except httpx.PoolTimeout as exc:
            raise ProviderError(
                f"Ollama request failed (pool exhausted): {exc}"
            ) from exc
        except httpx.WriteTimeout as exc:
            raise ProviderError(
                f"Ollama request failed (write timeout): {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Ollama request failed: {exc}"
            ) from exc

        if response.status_code >= 500:
            raise ProviderError(
                f"Ollama server error {response.status_code}: {response.text[:200]}"
            )
        if not response.is_success:
            raise ProviderResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"Ollama returned non-JSON response: {exc}. Body: {response.text[:200]}"
            ) from exc

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single response using the Ollama /api/chat endpoint."""
        model = request.model or self.default_model
        if not model:
            raise ConfigurationError(
                "No model specified and no default model is configured"
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_dict(m) for m in request.messages],
            "stream": False,
        }
        keep_alive = getattr(request, "keep_alive", None) or "60m"
        payload["keep_alive"] = keep_alive

        options = payload.setdefault("options", {})
        num_ctx = getattr(request, "num_ctx", None)
        options.setdefault("num_ctx", num_ctx if num_ctx is not None else 16384)
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = request.stop
        # Control thinking mode for models like qwen3 that support it.
        # Set at the top level (not inside options) per Ollama API spec.
        think = getattr(request, "think", None)
        if think is not None:
            payload["think"] = think

        start = time.monotonic()
        try:
            data = await self._post("/api/chat", payload)
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            _logger.info(
                "ollama.generate  model=%s  latency_ms=%.1f",
                model,
                latency_ms,
            )

        raw_content = _get_in(data, ("message", "content"), "")
        if not raw_content and _get_in(data, ("message", "thinking"), ""):
            raw_content = _get_in(data, ("message", "thinking"), "")

        return GenerationResponse(
            content=raw_content,
            model=data.get("model", model),
            raw=data,
            finish_reason=data.get("done_reason"),
        )

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama using SSE /api/chat."""
        model = request.model or self.default_model
        if not model:
            raise ConfigurationError(
                "No model specified and no default model is configured"
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_dict(m) for m in request.messages],
            "stream": True,
        }
        keep_alive = getattr(request, "keep_alive", None) or "60m"
        payload["keep_alive"] = keep_alive

        stream_options = payload.setdefault("options", {})
        num_ctx = getattr(request, "num_ctx", None)
        stream_options.setdefault("num_ctx", num_ctx if num_ctx is not None else 16384)
        if request.temperature is not None:
            stream_options["temperature"] = request.temperature
        if request.top_p is not None:
            stream_options["top_p"] = request.top_p
        if request.max_tokens is not None:
            stream_options["num_predict"] = request.max_tokens
        if request.stop:
            stream_options["stop"] = request.stop
        # Control thinking mode for models like qwen3 that support it.
        # Set at the top level (not inside options) per Ollama API spec.
        think = getattr(request, "think", None)
        if think is not None:
            payload["think"] = think

        try:
            async with self._client.stream(
                "POST",
                self._url("/api/chat"),
                json=payload,
            ) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(model)
                if response.status_code >= 500:
                    text = await response.aread()
                    raise ProviderError(
                        f"Ollama server error {response.status_code}: {text[:200]}"
                    )
                if not response.is_success:
                    text = await response.aread()
                    raise ProviderResponseError(
                        f"Ollama returned HTTP {response.status_code}: {text[:200]}"
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise ProviderError(f"Ollama stream error: {chunk['error']}")
                    msg = chunk.get("message", {})
                    thinking_token = msg.get("thinking", "")
                    content_token = msg.get("content", "") or chunk.get("response", "")
                    if chunk.get("done"):
                        self.last_metadata = chunk
                    if thinking_token:
                        yield StreamChunk(thinking=thinking_token, is_thinking=True)
                    if content_token:
                        yield StreamChunk(content=content_token, is_thinking=False)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self._config.base_url}: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ProviderTimeoutError(
                f"Ollama stream timed out after "
                f"{self._config.request_timeout_seconds}s"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama stream timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama stream failed: {exc}") from exc

    async def health(self) -> ProviderHealth:
        """Check whether the Ollama server is reachable via /api/tags."""
        try:
            data = await self._get("/api/tags")
            models_raw: list[dict] = data.get("models") or []
            models = [
                ModelInfo(
                    name=m.get("name", ""),
                    size_bytes=m.get("size"),
                    modified_at=m.get("modified_at"),
                    family=m.get("details", {}).get("family"),
                    parameter_size=m.get("details", {}).get("parameter_size"),
                    quantization_level=m.get("details", {}).get("quantization_level"),
                )
                for m in models_raw
                if m.get("name")
            ]
            return ProviderHealth(
                provider=self.name,
                reachable=True,
                model_count=len(models),
                raw={"models": [m.name for m in models]},
            )
        except ProviderUnavailableError as exc:
            return ProviderHealth(
                provider=self.name,
                reachable=False,
                model_count=0,
                error=f"Connection refused: {exc}",
            )
        except ProviderTimeoutError as exc:
            return ProviderHealth(
                provider=self.name,
                reachable=False,
                model_count=0,
                error=f"Timeout: {exc}",
            )
        except LLMError as exc:
            return ProviderHealth(
                provider=self.name,
                reachable=False,
                model_count=0,
                error=str(exc),
            )

    async def list_models(self) -> list[ModelInfo]:
        """Return the list of models available on the Ollama server."""
        data = await self._get("/api/tags")
        models_raw: list[dict] = data.get("models") or []
        return [
            ModelInfo(
                name=m.get("name", ""),
                size_bytes=m.get("size"),
                modified_at=m.get("modified_at"),
                family=m.get("details", {}).get("family"),
                parameter_size=m.get("details", {}).get("parameter_size"),
                quantization_level=m.get("details", {}).get("quantization_level"),
            )
            for m in models_raw
            if m.get("name")
        ]


def _message_to_dict(msg: ChatMessage) -> dict[str, str]:
    return {"role": msg.role.value, "content": msg.content}


def _get_in(data: dict, path: tuple[str, ...], default: Any = None) -> Any:
    """Safely navigate a nested dict."""
    for key in path:
        if isinstance(data, dict):
            data = data.get(key)
            if data is None:
                return default
        else:
            return default
    return data if data is not None else default