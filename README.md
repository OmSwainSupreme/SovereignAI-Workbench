# SovereignAI Workbench

An **air-gapped, self-hosted, multimodal agentic AI platform** for confidential enterprise environments.

SovereignAI Workbench enables organizations to run local open-weight LLMs, orchestrate agentic multi-step workflows, process documents with OCR/vision, manage local RAG knowledge bases, execute sandboxed code, and enforce policy controls — all without any external AI or API calls.

---

## Status

> **Implementation complete.** All core subsystems are implemented and passing 793 tests. The platform is preserved on the `phase-9-frontend` branch as a historical reference implementation.

---

## What Is Built

| Subsystem | Location | Status |
|---|---|---|
| Model Gateway (Ollama provider) | `core/llm/` | ✅ Complete |
| Model Router (capability-based) | `core/routing/` | ✅ Complete |
| Policy Engine (deny-by-default) | `core/security/` | ✅ Complete |
| Tool Registry | `core/tools/registry.py` | ✅ Complete |
| Secure Workspace & File Tools | `core/tools/workspace.py`, `file_tools.py` | ✅ Complete |
| Document Tools (DOCX/TXT/JSON) | `core/tools/document_tools.py` | ✅ Complete |
| Local RAG / Knowledge Base | `core/rag/` | ✅ Complete |
| OCR & Vision Processing | `core/vision/` | ✅ Complete |
| Docker Code Sandbox | `core/sandbox/` | ✅ Complete |
| Agent Runtime (ReAct loop) | `core/agent/` | ✅ Complete |
| Intent Classifier | `core/agent/intent.py` | ✅ Complete |
| Thinking / Token Streaming | `core/llm/types.py`, `core/agent/` | ✅ Complete |
| Privacy-Aware Audit Logging | `core/audit_logger.py` | ✅ Complete |
| REST API (FastAPI v1) | `backend/app/api/v1/` | ✅ Complete |
| Frontend (Vite + React) | `frontend/` | ✅ Complete |

---

## Architecture

```
frontend/               Vite + React + TanStack Router UI
backend/
  app/
    api/v1/             REST endpoints (tasks, files, knowledge, vision, code, activity)
    core/               Settings, logging
    models/             Pydantic schemas
  tests/                793 passing tests
core/
  agent/                Agent runtime, planner, verifier, intent classifier
  llm/                  Model gateway, Ollama provider, streaming
  routing/              Capability-based model router
  security/             Policy engine (ALLOW / DENY / REQUIRE_APPROVAL)
  tools/                Tool registry, workspace, file tools, document tools
  rag/                  Chunker, embeddings, vector store, retriever, document parser
  vision/               OCR, vision analysis, image loader, RAG bridge
  sandbox/              Docker-based isolated code execution
  audit_logger.py       Privacy-aware audit logging
config/
  models.yaml           Model registry (logical names → Ollama models)
  policy.yaml           Policy rules
sandbox/
  Dockerfile            Sandboxed Python execution container
  entrypoint.py         Process group supervisor
workspace/              Default agent workspace directory
```

---

## Core Principles

- **Local-first** — All LLM traffic is loopback-only. No external API calls are ever made.
- **Air-gapped capable** — No cloud SDKs, no telemetry, no automatic model downloads by agents.
- **Model-agnostic** — Provider abstraction decouples agents from any specific model or runtime.
- **Deny-by-default** — Policy engine evaluates every tool call before execution. Unknown actions are denied.
- **Workspace isolation** — All file access is path-contained. No `../` traversal, no symlink escapes.
- **Sandbox isolation** — Code execution runs in a Docker container with dropped capabilities, no network, resource limits.
- **No sensitive logging** — OCR text, vision descriptions, prompts, and image bytes are never written to logs.

---

## Quick Start

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) running locally with at least one model pulled
- Docker (for sandbox code execution)
- Node.js 18+ (for the frontend)

### Backend

```powershell
# From the repository root — create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
cd backend
pip install -r requirements.txt -r requirements-dev.txt
cd ..

# Start the backend
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

API available at:
- `http://127.0.0.1:8000/docs` — Swagger UI
- `http://127.0.0.1:8000/api/v1/models/status` — Model status

### Frontend

```powershell
cd frontend
npm install    # or: bun install
npm run dev    # or: bun run dev
```

Frontend available at `http://localhost:5173`.

### Recommended Local Models

```powershell
ollama pull qwen3:4b          # General reasoning
ollama pull qwen2.5-coder:3b  # Code generation
ollama pull qwen2.5vl:3b      # Vision / OCR
```

---

## Running Tests

```powershell
# From the repository root (requires activated venv)
pytest backend/tests -q
```

**Result: 793 passed, 11 skipped** (skipped tests require a live Docker daemon or live Ollama).

No live Ollama or Docker instance is required to run the test suite. All provider interactions use fake transports.

---

## Configuration

Copy `.env.example` to `.env` in the repository root and adjust as needed.

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `WORKSPACE_PATH` | `workspace` | Root directory for agent file operations |
| `AGENT__MAX_ITERATIONS` | `10` | Max agent loop iterations per run |
| `AGENT__MAX_TOOL_CALLS` | `30` | Max tool calls per agent run |
| `LOG_LEVEL` | `info` | Logging verbosity |

No secrets are embedded in the codebase.

---

## Docker Compose

```powershell
docker-compose up --build
```

The backend container expects Ollama reachable at `OLLAMA_BASE_URL`. Add an Ollama sidecar to the compose stack for a fully containerised deployment.

---

## Repository Notes

This repository is the **legacy SovereignAI Workbench** (v0.8.0). It is preserved as a historical reference implementation on the `phase-9-frontend` branch and pushed to `Sovereign-AI-2.0` for archival.

The successor project is **SovereignAI 2.0** — a clean-slate, production-grade redesign with formal phase specifications, typed contracts, persistence layer, and independent QA.

---

## License

MIT License — see `LICENSE` file for details.
