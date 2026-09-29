# SovereignAI Workbench

An **air-gapped, self-hosted, multimodal agentic AI platform** for confidential enterprise environments.

SovereignAI Workbench enables organizations to run local open-weight LLMs, orchestrate agentic multi-step workflows, process documents with OCR/vision, manage local RAG knowledge bases, execute sandboxed code, and enforce policy controls — all without any external AI or API calls.

## Project Purpose

The goal is a complete, sovereign AI workbench that never phones home:

- **Local open-weight LLMs** — run GGUF, GPTQ, AWQ models via a pluggable model abstraction layer
- **Model routing** — direct requests to the right local model based on task or configuration
- **Agentic workflows** — multi-step task orchestration with tool use and memory
- **OCR and vision** — extract and understand content from images and scanned documents
- **Local RAG** — knowledge-base ingestion, embedding, and retrieval entirely on-premise
- **Sandboxed code execution** — run generated or user-supplied code safely and in isolation
- **DOCX generation** — produce Word documents from structured data
- **Policy engine** — rule-based enforcement of organizational constraints
- **Audit logging** — complete, tamper-evident event records for every operation
- **Sovereignty monitoring** — detect and block unexpected outbound connections

## Current Phase

> **Phase 5C — OCR & Vision Processing.** A fully local OCR + Vision subsystem is implemented. Images are loaded securely from the workspace, text is extracted via OCR, and images are analysed via vision. Two provider families ship: deterministic `FakeOCRProvider`/`FakeVisionProvider` for development and tests, and real `OllamaOCRProvider`/`OllamaVisionProvider` that talk to a local Ollama vision model (e.g. `qwen2.5vl:3b`) over loopback HTTP for production. The `ocr_image` and `analyze_image` tools are integrated into the Agent's ToolRegistry. OCR output can be converted to a RAG `Document` via `ocr_to_document`. No cloud APIs are used.

## Completed Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 2B | Model Gateway & Ollama Provider | ✅ Complete |
| 3 | Model Router | ✅ Complete |
| 4 | Agent Runtime | ✅ Complete |
| 5A | File Tools & Workspace Security | ✅ Complete |
| 5B | Local RAG / Knowledge Base | ✅ Complete |
| 5C | OCR & Vision Processing | ✅ Complete |
| 5D | Docker Sandbox & Security Hardening | ✅ Complete |

## Architecture

### High-level repository layout

```
sovereign-ai/
├── backend/          FastAPI REST API
│   └── app/
│       ├── api/
│       │   ├── routes.py         Top-level (/, /health, /info)
│       │   └── v1/
│       │       └── models_router.py   /api/v1/models/status
│       ├── core/
│       │   ├── config.py        Application settings (incl. LLM__* env vars)
│       │   └── logging.py       Structured logging configuration
│       ├── services/
│       │   └── model_service.py  Application-level gateway wrapper
│       └── models/schemas.py    Pydantic request/response models
├── core/             Framework-agnostic domain logic
│   └── llm/                       Provider-agnostic model layer
│       ├── types.py              ChatMessage, GenerationRequest, …
│       ├── errors.py             LLMError hierarchy
│       ├── registry.py           ProviderRegistry
│       ├── gateway.py            ModelGateway
│       └── providers/
│           ├── base.py           BaseProvider ABC
│           └── ollama.py         OllamaProvider (localhost only)
├── frontend/         Web UI (future)
├── config/           Configuration schemas and environment mappings (future)
├── data/             Local databases, model weights, cached files (gitignored)
├── sandbox/          Isolated runtime for code execution (future)
├── scripts/          Deployment and operational utility scripts (future)
└── tests/            Integration and end-to-end tests (future)
```

### Model Gateway architecture

```
            ┌───────────────────────┐
            │   Agent  (future)     │
            └──────────┬────────────┘
                       │  uses
            ┌──────────▼────────────┐
            │   ModelGateway        │   core/llm/gateway.py
            │   (provider-agnostic) │
            └──────────┬────────────┘
                       │  selects
            ┌──────────▼────────────┐
            │   ProviderRegistry    │   core/llm/registry.py
            └──────────┬────────────┘
                       │  instantiates
            ┌──────────▼────────────┐
            │   BaseProvider (ABC)  │   core/llm/providers/base.py
            │      ▲                │
            │      │ implements     │
            │  ┌───┴────────────┐   │
            │  │ OllamaProvider  │   │   core/llm/providers/ollama.py
            │  └────────────────┘   │
            └──────────┬────────────┘
                       │  httpx (loopback only)
            ┌──────────▼────────────┐
            │  http://127.0.0.1:    │
            │       11434           │   local Ollama server only
            └───────────────────────┘
```

Application code (FastAPI routes, future agents) **never imports `OllamaProvider`** directly. All model calls go through the gateway, which selects the configured provider. Adding a new local provider (llama.cpp, vLLM, etc.) is a one-line registration in `ProviderRegistry` plus a new `core/llm/providers/<name>.py` module.

## Current Local Provider: Ollama

The current local provider is [Ollama](https://ollama.com/). Ollama is an open-source runtime that serves open-weight LLMs (Llama 3, Mistral, Phi-3, Qwen 2, etc.) over a local HTTP API. The SovereignAI Workbench backend communicates with Ollama using a pure-HTTP client (`httpx`) — no proprietary SDK is involved.

### Installing Ollama

Ollama is installed **separately** from this application. It is NOT bundled with the Workbench and the application does not download models automatically.

```powershell
# Windows (PowerShell) — using the official installer:
# Download from https://ollama.com/download and run the installer.

# Or via winget:
winget install Ollama.Ollama

# Or via the MSI:
# https://ollama.com/download/OllamaSetup.exe
```

On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pulling a model

After installing Ollama, pull a model you want to use:

```powershell
# Start the Ollama server (typically runs as a system service on Windows)
ollama serve

# In another terminal, pull a model
ollama pull llama3
ollama pull phi3
ollama pull mistral
```

> The SovereignAI Workbench will not download models on your behalf. Choose models sized appropriately for your hardware. On the development machine (Ryzen 3 3250U, 6 GB RAM), small 1B–3B parameter models in Q4 or Q5 quantisation are recommended for testing.

### Verifying Ollama is running

```powershell
curl http://127.0.0.1:11434/api/tags
```

You should see a JSON list of available models.

## Configuring the Local Provider

The model gateway is configured via environment variables. Defaults are shown in `.env.example`.

| Variable | Default | Description |
|---|---|---|
| `LLM__PROVIDER` | `ollama` | Provider name (must be registered in `ProviderRegistry`) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Base URL of the local Ollama server. **Must be loopback.** |
| `OLLAMA_DEFAULT_MODEL` | _(empty)_ | Model to use when a request does not specify one |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `120` | HTTP request timeout |

> The provider **refuses to start** if `OLLAMA_BASE_URL` does not point at a loopback host (`127.0.0.1`, `::1`, or `localhost`). This is a deliberate security guarantee: even if the environment is misconfigured, the application cannot accidentally route model requests to an external host.

## Verifying Connectivity

A diagnostic endpoint is available to confirm the local provider is reachable:

```powershell
curl http://127.0.0.1:8000/api/v1/models/status
```

A reachable provider returns HTTP 200:

```json
{
  "provider": "ollama",
  "reachable": true,
  "model_count": 3,
  "error": null
}
```

An unreachable provider returns HTTP 503 with the connection error:

```json
{
  "provider": "ollama",
  "reachable": false,
  "model_count": 0,
  "error": "Connection refused: ..."
}
```

The endpoint only calls `GET /api/tags` on the configured provider — it does NOT call any generative endpoint, does NOT transmit prompts or document content, and does NOT expose any administrative interface.

## No Cloud Model Provider

> This phase does not support any cloud model provider. There is no OpenAI, Anthropic, Gemini, Mistral Cloud, AWS Bedrock, or other SaaS integration. The only supported provider is a **locally running Ollama server**, and the application refuses to start against any non-loopback endpoint.

This is by design and is core to the air-gapped, sovereignty-first architecture of the platform.

## Local RAG / Knowledge Base (Phase 5B)

The RAG subsystem provides document ingestion, chunking, embedding, and semantic search entirely offline. No cloud APIs, no external embedding services, no telemetry.

### Architecture

```
Document (workspace-relative path)
        ↓
DocumentParser (PlainTextParser for .txt/.md)
        ↓
TextChunker (DeterministicChunker with configurable size/overlap)
        ↓
EmbeddingProvider (FakeEmbeddingProvider in dev; swappable)
        ↓
VectorStore (SimpleVectorStore — in-memory, no external DB)
        ↓
Semantic Search → SearchResult (ranked, with provenance)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `KnowledgeBase` | `core/rag/knowledge_base.py` | Top-level ingest/search orchestrator |
| `DocumentParser` | `core/rag/document_parser.py` | Converts raw text to `Document` |
| `TextChunker` | `core/rag/chunker.py` | Splits documents into `DocumentChunk` |
| `EmbeddingProvider` | `core/rag/embedding.py` | Text → vector (abstract; `FakeEmbeddingProvider` impl) |
| `VectorStore` | `core/rag/vector_store.py` | Stores vectors, answers similarity queries |
| `Retriever` | `core/rag/retriever.py` | Bridges embedding + vector store |
| `search_knowledge_base` | `core/rag/tools.py` | Agent tool (via ToolRegistry) |

### Supported Document Types

Currently **text-only**:

- `.txt` — plain text
- `.md` — Markdown

**Not yet supported** (future phases):
- Scanned PDFs (PDF rendering not implemented; OCR of rendered pages is available via Phase 5C)
- DOCX, DOC
- Images (direct image-to-RAG ingestion; `ocr_to_document` exists for OCR output)

### Configuration

The RAG subsystem is configured via `config/rag.yaml`:

```yaml
knowledge_workspace_dir: data/knowledge  # Must be inside workspace
chunk_size: 500                        # Characters per chunk
overlap: 50                            # Overlapping characters
top_k: 5                               # Default result count
min_score: 0.0                         # Minimum cosine similarity
embedding_provider: fake               # "fake" only in this phase
embedding_dimension: 128               # Vector dimension
max_file_bytes: 10485760               # 10 MiB max file size
```

### Usage Example

```python
from core.rag import create_knowledge_base, register_rag_tools
from core.tools import Workspace
from core.agent.registry import DefaultToolRegistry

# Create the knowledge base
workspace = Workspace(root_path="data/knowledge")
kb = create_knowledge_base(
    workspace=workspace,
    chunk_size=500,
    overlap=50,
)

# Register the search tool with the agent
registry = DefaultToolRegistry()
register_rag_tools(registry, kb)

# Ingest documents
kb.ingest("manuals/pump-operations.txt")
kb.ingest("sops/safety-checklist.md")

# Search
results = kb.search("centrifugal pump dry-run limitations", top_k=3)
for r in results.results:
    print(f"[{r.score:.2f}] {r.chunk.metadata.filename}: {r.chunk.text[:100]}...")

# Agent can now call search_knowledge_base tool
```

### Security Properties

- **Workspace boundary enforced**: All file access goes through `Workspace.resolve()` — no arbitrary paths, no `../` traversal, no symlink escapes
- **No external network access**: Zero HTTP calls during ingestion or search
- **No sensitive content logging**: Document text, chunk content, queries, and embeddings are never written to logs
- **Idempotent ingestion**: Re-ingesting the same file replaces old chunks (upsert semantics)

### No External Dependencies

The RAG subsystem has **no external dependencies**:
- No LangChain, LangGraph, or similar frameworks
- No OpenAI/Anthropic/Google/Cohere embeddings
- No Pinecone/Weaviate/Qdrant cloud
- No Hugging Face hosted inference
- No numpy (pure Python vector operations)

### Running Tests

```powershell
# Run all tests
pytest

# Run RAG-specific tests only
pytest backend/tests/test_rag.py -v

# Run with strict markers
pytest --strict-markers -q
```

**Current test coverage**: 78 RAG-specific tests covering parsing, chunking, embeddings, vector store, knowledge base, agent tool integration, security boundary, and determinism.

## OCR & Vision Processing (Phase 5C)

The OCR/Vision subsystem extracts text from images and analyses image content entirely offline. It reuses the same workspace security boundary as the file tools.

### Architecture

```
Image (workspace-relative path)
        ↓
ImageLoader (workspace-enforced path + content-type checks)
        ↓
OCRProvider / VisionProvider (Fake* in dev, Ollama* in production)
        ↓
OCRResult / VisionResult (structured, no raw bytes)
        ↓
ocr_to_document → RAG Document  |  Agent tools (ocr_image, analyze_image)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ImageLoader` | `core/vision/image_loader.py` | Loads images from the workspace with security guarantees |
| `OCRProvider` | `core/vision/ocr.py` | Abstract OCR interface; `FakeOCRProvider` implementation |
| `VisionProvider` | `core/vision/vision.py` | Abstract vision interface; `FakeVisionProvider` implementation |
| `OllamaOCRProvider` | `core/vision/ollama_ocr.py` | Real OCR via local Ollama vision model (`/api/generate`) |
| `OllamaVisionProvider` | `core/vision/ollama_vision.py` | Real vision analysis via local Ollama vision model |
| `ocr_to_document` | `core/vision/rag_bridge.py` | Converts OCR output to a RAG `Document` for ingestion |
| `ocr_image` / `analyze_image` | `core/vision/tools.py` | Agent tools (via ToolRegistry) |

### Supported Image Types

- `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`

### Providers

Two provider families are interchangeable behind the same interfaces:

- **Fake** (`FakeOCRProvider`, `FakeVisionProvider`) — deterministic placeholder output. Used in tests and development without a model. The default.
- **Ollama** (`OllamaOCRProvider`, `OllamaVisionProvider`) — real OCR/vision via a local Ollama server running a vision-capable model such as `qwen2.5vl:3b`. **Loopback-only** at construction time.

```python
from core.vision import (
    OllamaOCRProvider,
    OllamaVisionProvider,
    ImageLoader,
    register_vision_tools,
)
from core.tools import Workspace
from core.agent import DefaultToolRegistry

workspace = Workspace(root_path="data/workspace")
loader = ImageLoader(workspace)

# Real providers backed by a local Ollama vision model.
ocr = OllamaOCRProvider(base_url="http://127.0.0.1:11434", default_model="qwen2.5vl:3b")
vision = OllamaVisionProvider(base_url="http://127.0.0.1:11434", default_model="qwen2.5vl:3b")

# Register the agent tools.
registry = DefaultToolRegistry()
register_vision_tools(registry, ocr, vision, loader)

# Direct use.
image = loader.load("scans/invoice.png")
result = ocr.recognize(image)
print(result.full_text)
analysis = vision.analyze(image, "Describe this image")
print(analysis.description)
```

### Security Properties

- **Workspace boundary enforced**: All file access goes through `Workspace.resolve()` — no arbitrary paths, no `../` traversal, no symlink escapes (delegated to `ImageLoader`).
- **No external network access**: Ollama providers communicate loopback-only; fake providers make zero HTTP calls.
- **No sensitive content logging**: OCR text, vision descriptions, image bytes, and prompts are never written to logs.
- **Loopback enforcement**: Ollama providers refuse to construct against any non-loopback base URL, mirroring the model gateway guarantee.

### Agent Integration

Both tools are registered through `register_vision_tools` and become available to the Agent's `ToolRegistry`:

- `ocr_image` — `{"path": "scans/invoice.png"}` → extracted text + confidence
- `analyze_image` — `{"path": "images/plant.jpg", "prompt": "Describe..."}` → description + tags

OCR output can be fed to the knowledge base via `ocr_to_document` (`core/vision/rag_bridge.py`), so scanned documents become searchable in RAG.

### Running Tests

```powershell
# Run vision/OCR-specific tests only
pytest backend/tests/test_vision.py backend/tests/test_ollama_vision_provider.py -v
```

**Current test coverage**: 81 vision/OCR-specific tests covering provider interfaces, fake provider determinism, Ollama provider behaviour (via mocked HTTP transport), image loading, agent tool integration, path security, logging hygiene, RAG compatibility, and vision routing.

## Development Prerequisites

- **Python 3.11** — see [python.org/downloads](https://www.python.org/downloads/)
- **pip** — comes bundled with Python
- **Ollama** (optional for development without model calls) — see above
- **Docker & Docker Compose** — for containerized deployment (future)
- **Git** — for version control

## Virtual Environment Setup

Create and activate a virtual environment from the repository root:

```powershell
# Create the venv
python -m venv .venv

# Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate it (Cmd)
.venv\Scripts\activate.bat

# Activate it (Git Bash / WSL)
source .venv/Scripts/activate
```

> The `.venv` directory is already excluded from Git in `.gitignore`.

## Backend Installation

```powershell
# Install runtime and dev dependencies
cd backend
pip install -r requirements.txt -r requirements-dev.txt
```

Or using the editable install with dev extras:

```powershell
pip install -e backend/.[dev]
```

## Backend Startup

```powershell
# Run from the repository root (module path is required for absolute imports)
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# The API will be available at:
#   http://127.0.0.1:8000
#   http://127.0.0.1:8000/docs  (Swagger UI)
#   http://127.0.0.1:8000/redoc (ReDoc)
#   http://127.0.0.1:8000/api/v1/models/status
```

> `--reload` enables auto-reload on code changes — use only during development.

## Testing

```powershell
# Run from the repository root
pytest

# Or with verbose output
pytest -v

# Run with coverage
pytest --cov=backend --cov-report=term-missing
```

> Tests do not require a running Ollama server. All provider tests use a fake `httpx.MockTransport` to simulate Ollama behaviour.

## Configuration

All settings are loaded from environment variables with sensible defaults. Copy `.env.example` from the repository root to a `.env` file in the `backend/` directory and adjust as needed. **No secrets are embedded in the codebase.**

## Air-Gapped Deployment

The application is designed to be deployable in fully air-gapped environments:

- All LLM traffic is loopback-only (enforced at provider construction time).
- No external URL is ever contacted by the application code.
- The `httpx` library does not perform any automatic telemetry.
- No SDKs that phone home are used (no `ollama-python` SDK, no OpenAI SDK, no LangChain telemetry).

When deploying in an air-gapped environment, you will need to:

1. Pre-install Ollama (or your preferred local LLM runtime) on the same host.
2. Pre-download model weights via `ollama pull`.
3. Pre-stage Python wheels on a local pip mirror and install from there.
4. Configure `OLLAMA_BASE_URL` to point at the local Ollama instance.

## Current Project Limitations

The following capabilities do not yet exist — they will be built in subsequent phases:

- No DOCX generation
- No policy engine
- No audit logging
- No authentication or authorization
- No frontend UI
- No database connection
- No automatic model download
- No support for cloud model providers (by design)
- RAG ingestion is text-only (`.txt`, `.md`); scanned PDFs, DOCX, and OCR are out of scope for this phase
- RAG embedding uses a deterministic FakeEmbeddingProvider in development; production-grade embedding models are a future configuration option

## Docker Compose (Containerized Deployment)

```powershell
docker-compose up --build
```

See `docker-compose.yml` for the full service definitions. The backend container expects an Ollama daemon to be reachable on the configured `OLLAMA_BASE_URL`; in a real deployment an Ollama sidecar or systemd-managed Ollama would be added to the compose stack. This is left to the deployment phase.

## License

MIT License — see `LICENSE` file for details.
