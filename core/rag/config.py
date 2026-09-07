"""Configuration loader for the local RAG subsystem.

This module loads the local RAG configuration from a YAML or JSON file
following the same convention as :mod:`core.routing.registry`. The
config declares:

* the controlled knowledge workspace directory
* chunking parameters
* the default embedding provider
* default search parameters

The loader is local-only: it never makes a network call. It validates
the values at load time and raises a clear error on misconfiguration.

Example::

    from core.rag.config import load_rag_config

    cfg = load_rag_config("config/rag.yaml")
    kb = create_knowledge_base(
        workspace=Workspace(root_path=cfg.knowledge_workspace_dir),
        embedding_provider=build_embedding_provider(cfg),
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Union

from core.rag.errors import RAGConfigurationError


# Default values match the bundled ``config/rag.yaml``. They are kept in
# sync with that file; the loader applies them when keys are absent.
_DEFAULTS: dict[str, object] = {
    "knowledge_workspace_dir": "data/knowledge",
    "chunk_size": 500,
    "overlap": 50,
    "top_k": 5,
    "min_score": 0.0,
    "embedding_provider": "fake",
    "embedding_dimension": 128,
    "max_file_bytes": 10 * 1024 * 1024,
}

# Embedding providers that can be referenced by name in config. The
# local-only contract is preserved: no external endpoints.
_VALID_PROVIDERS = frozenset({"fake", "hash"})


PathLike = Union[str, Path]


@dataclass(frozen=True)
class RAGConfig:
    """A parsed local-RAG configuration."""

    knowledge_workspace_dir: str
    chunk_size: int
    overlap: int
    top_k: int
    min_score: float
    embedding_provider: str
    embedding_dimension: int
    max_file_bytes: int
    source_path: str = ""

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise RAGConfigurationError("chunk_size must be at least 1")
        if self.overlap < 0:
            raise RAGConfigurationError("overlap must not be negative")
        if self.overlap >= self.chunk_size:
            raise RAGConfigurationError("overlap must be smaller than chunk_size")
        if self.top_k < 1:
            raise RAGConfigurationError("top_k must be at least 1")
        if not 0.0 <= self.min_score <= 1.0:
            raise RAGConfigurationError("min_score must be between 0.0 and 1.0")
        if self.embedding_provider not in _VALID_PROVIDERS:
            raise RAGConfigurationError(
                f"embedding_provider {self.embedding_provider!r} is not supported; "
                f"valid options: {sorted(_VALID_PROVIDERS)}"
            )
        if self.embedding_dimension < 1:
            raise RAGConfigurationError("embedding_dimension must be at least 1")
        if self.max_file_bytes < 1:
            raise RAGConfigurationError("max_file_bytes must be at least 1")


def load_rag_config(path: PathLike) -> RAGConfig:
    """Load a :class:`RAGConfig` from a YAML or JSON file.

    The file format is selected by extension: ``.yaml``/``.yml`` use YAML
    (requires PyYAML); anything else falls back to JSON.

    Args:
        path: Path to the configuration file.

    Returns:
        A :class:`RAGConfig` populated from the file with defaults
        applied for missing keys.

    Raises:
        RAGConfigurationError: if the file is missing, malformed, or
            contains invalid values.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise RAGConfigurationError(
            f"RAG configuration file not found: {file_path}"
        )

    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RAGConfigurationError(
                f"YAML RAG config requested but PyYAML is not installed: {exc}"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise RAGConfigurationError(
            f"RAG config root must be a mapping, got {type(data).__name__}"
        )

    merged: dict[str, object] = dict(_DEFAULTS)
    for key, value in data.items():
        merged[key] = value

    return _build_config(merged, source_path=str(file_path))


def load_rag_config_from_mapping(
    data: Mapping[str, object],
    *,
    source_path: str = "",
) -> RAGConfig:
    """Load a :class:`RAGConfig` from an in-memory mapping.

    Useful for tests and for callers that build the config dynamically.
    Unknown keys are ignored.
    """
    merged: dict[str, object] = dict(_DEFAULTS)
    for key, value in data.items():
        merged[key] = value
    return _build_config(merged, source_path=source_path)


def build_embedding_provider(config: RAGConfig) -> "EmbeddingProvider":
    """Build an embedding provider from a :class:`RAGConfig`.

    This is the factory that wires a config file to a concrete provider.
    It is local-only: no network calls, no model downloads. The
    available providers are:

    * ``fake`` — :class:`FakeEmbeddingProvider`, deterministic, used
      for tests and low-resource development machines.
    * ``hash`` — placeholder for a future bag-of-words hashing
      provider; not implemented in this phase. Asking for it raises
      a clear :class:`RAGConfigurationError`.

    Args:
        config: A loaded :class:`RAGConfig`.

    Returns:
        A concrete :class:`EmbeddingProvider`.

    Raises:
        RAGConfigurationError: if the configured provider name is not
            supported.
    """
    # Imported lazily to avoid a circular import with embedding.py.
    from core.rag.embedding import FakeEmbeddingProvider

    if config.embedding_provider == "fake":
        return FakeEmbeddingProvider(dimension=config.embedding_dimension)
    raise RAGConfigurationError(
        f"embedding_provider {config.embedding_provider!r} is not implemented in this phase"
    )


def _build_config(merged: dict[str, object], source_path: str) -> RAGConfig:
    try:
        return RAGConfig(
            knowledge_workspace_dir=str(merged["knowledge_workspace_dir"]),
            chunk_size=int(merged["chunk_size"]),  # type: ignore[arg-type]
            overlap=int(merged["overlap"]),  # type: ignore[arg-type]
            top_k=int(merged["top_k"]),  # type: ignore[arg-type]
            min_score=float(merged["min_score"]),  # type: ignore[arg-type]
            embedding_provider=str(merged["embedding_provider"]),
            embedding_dimension=int(merged["embedding_dimension"]),  # type: ignore[arg-type]
            max_file_bytes=int(merged["max_file_bytes"]),  # type: ignore[arg-type]
            source_path=source_path,
        )
    except (TypeError, ValueError) as exc:
        raise RAGConfigurationError(
            f"Invalid RAG configuration: {exc}"
        ) from exc
