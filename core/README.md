# SovereignAI Workbench — Core

Framework-agnostic domain logic, shared utilities, and pluggable component interfaces for SovereignAI Workbench.

## Purpose

This directory holds the business logic that is independent of any particular web framework, database, or external service. The `backend` layer calls into `core`; `core` never imports from `backend` or `frontend`.

## Implemented Modules

| Module | Description |
|---|---|
| `llm/` | Model gateway, provider abstraction, and Ollama implementation (Phase 2B) |

## Planned Modules

| Module | Description |
|---|---|
| `agent/` | Task orchestration, multi-step workflow engine (future) |
| `rag/` | Document chunking, embedding, vector storage, retrieval (future) |
| `ocr/` | Document OCR and image understanding (future) |
| `policy/` | Rule evaluation, compliance checks (future) |
| `audit/` | Structured event logging and export (future) |
| `security/` | Secrets handling, input sanitization, network monitoring (future) |

## llm/ Module (Phase 2B — Implemented)

The `llm/` module provides a provider-agnostic interface to local open-weight LLMs.

```
llm/
├── types.py           Public dataclasses: ChatMessage, GenerationRequest,
│                      GenerationResponse, ModelInfo, ProviderHealth, etc.
├── errors.py          Exception hierarchy (LLMError, ProviderUnavailableError, …)
├── registry.py        ProviderRegistry — maps provider name → factory
├── gateway.py         ModelGateway — application-level model access facade
└── providers/
    ├── base.py        BaseProvider ABC
    └── ollama.py      OllamaProvider (localhost only)
```

Key design principles:

- **No direct Ollama imports** in business logic. All model access goes through `ModelGateway`.
- **Provider isolation**: each provider lives in `providers/<name>.py`. Adding llama.cpp or vLLM is one new file plus a one-line registry call.
- **Loopback enforcement**: the Ollama provider refuses to start if `base_url` is not `127.0.0.1`, `::1`, or `localhost`.
- **No prompt/content logging**: structured logs include provider, model, latency, success/error only.

## Architectural Constraints

- `core` must **never** import from `backend`, `frontend`, or any external network service.
- All AI/LLM integrations must be pluggable via configuration — no hard-coded model choices.
- The module must be compatible with air-gapped, offline deployment (no outbound network calls).
- Code in `core` should be testable without spinning up the full FastAPI application.
