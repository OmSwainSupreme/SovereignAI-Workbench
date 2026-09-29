# SovereignAI Workbench — Backend

FastAPI-based REST API for SovereignAI Workbench.

## Purpose

The backend exposes HTTP endpoints consumed by the frontend and external clients.

In Phase 2B, the backend also serves as the host for the application-level
`ModelService`, which builds and exposes a `ModelGateway` over a configurable
local LLM provider (currently Ollama). Future phases will add agent, RAG, OCR,
and other capabilities on top of this foundation.

## Package Structure

```
backend/
├── app/
│   ├── main.py          FastAPI application entry point
│   ├── api/
│   │   ├── routes.py   Top-level routes (/, /health, /info)
│   │   └── v1/
│   │       └── models_router.py   /api/v1/models/status
│   ├── core/
│   │   ├── config.py    pydantic-settings — application + LLM configuration
│   │   └── logging.py   Structured logging configuration
│   ├── models/
│   │   └── schemas.py   Pydantic request/response models
│   └── services/
│       └── model_service.py   Application-level wrapper over ModelGateway
├── tests/
│   ├── test_health.py          Bootstrap endpoint tests
│   ├── test_llm_types.py       Tests for core.llm.types
│   ├── test_ollama_config.py   Tests for Ollama config / settings
│   ├── test_ollama_provider.py Tests for OllamaProvider (mocked HTTP)
│   ├── test_gateway.py         Tests for ModelGateway (fake provider)
│   └── test_model_status_endpoint.py   Tests for /api/v1/models/status
├── requirements.txt     Runtime dependencies
├── requirements-dev.txt Dev/test dependencies
├── pyproject.toml       Package metadata and tool configuration
├── Dockerfile           Container image definition
└── .dockerignore        Build context exclusions
```

## Development

### Prerequisites

- Python 3.11
- Virtual environment activated (see repository root `README.md`)

### Installation

```powershell
pip install -r requirements.txt -r requirements-dev.txt
```

### Running Locally

```powershell
# From the repository root — module path is required for absolute imports:
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# Or from within this directory:
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Running Tests

```powershell
# From the repository root:
pytest backend/tests/ -v

# Or from within this directory:
cd backend
pytest tests/ -v
```

### Configuration

Copy `.env.example` (repository root) to `backend/.env` and adjust values as needed. All settings have sensible defaults and can also be set via environment variables directly.

LLM configuration is set via:

- `LLM__PROVIDER` (e.g. `ollama`)
- `OLLAMA_BASE_URL` (must be loopback)
- `OLLAMA_DEFAULT_MODEL`
- `OLLAMA_REQUEST_TIMEOUT_SECONDS`

## Current Limitations

- Single LLM provider per process (no routing)
- No agent, RAG, OCR, vision, sandbox, DOCX, policy, audit
- No database
- No authentication
- No CORS configuration
- No rate limiting
- Cloud LLM providers are explicitly NOT supported

All of the above will be added in later phases.
