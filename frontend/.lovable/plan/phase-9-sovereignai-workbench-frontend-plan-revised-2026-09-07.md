# Phase 9 — SovereignAI Workbench Frontend Plan (revised)

A desktop-first, dark, local-feeling AI workstation UI that consumes the existing backend only. The backend stays the sole authority for authorization, policy, workspace security, sandboxing, provider security, and audit redaction. Nothing in this plan invents endpoints, schemas, auth flows, or a streaming protocol.

## 0. Confirmed vs unknown

**Confirmed (from the project brief):** the backend exists and is stable; it provides model gateway/router, agent runtime, workspace file tools, RAG, OCR/vision, sandboxed code execution, DOCX generation, a policy engine, and a privacy-aware audit logger. The frontend consumes it and must not replace any of it.

**Unknown until the backend team confirms:** every endpoint path, method, request/response/error schema, auth scheme, streaming transport, upload format, pagination, and every field name. Section 10 lists exactly what is needed. Until those arrive, no service call is written against a guessed shape — service modules are defined as typed interfaces with fixture adapters (Section 3) that are clearly non-production and removed as real contracts land.

## 1. Product surface

App shell: left rail (primary navigation) + contextual sidebar + main pane + optional right inspector.

- Workspace (default): chat/task interface with streaming, tool indicators, attachments
- History: task list, search and status filters
- Files: workspace browser, uploads, generated artifacts, DOCX download
- Knowledge: collections, query, retrieved sources
- Vision/OCR: upload, preview, processing state, structured results
- Code: run status and safe output
- Activity: audit timeline

Policy is not a page: allowed / denied / approval-required states render inline wherever they occur, plus a policy filter in Activity. There is no policy configuration UI.

## 2. Routes

```text
/                     -> redirect to /workspace
/workspace            -> new task
/workspace/$taskId    -> active task thread
/history
/files                -> browser; current folder held in a search param
/knowledge
/knowledge/$collectionId
/vision
/code
/code/$runId
/activity
```

Files uses `/files?path=…` rather than a splat path segment, and the path is only ever an opaque hint echoed back to the backend. A URL never grants access: every listing, read, upload, and download is authorized server-side, and an unauthorized or invalid path renders a neutral "not available" state without revealing filesystem structure.

Each route defines its own head() metadata. Shared chrome lives in the root layout.

## 3. Architecture

- TanStack Start + TanStack Router (file routes), TanStack Query for server state, Tailwind + shadcn components.
- Layers: `routes/` (thin) -> `features/<area>/` -> `services/api/<area>.ts` (typed client) -> `lib/http.ts` (fetch wrapper) -> backend. No component fetches directly.
- Single HTTP client: base URL from config, uniform error normalization, abort support, request-id propagation if the backend supplies one.
- Zod schemas per response, validated at the service boundary, written only once the real contract is supplied so drift fails loudly instead of rendering blanks.
- Contract gap strategy: each service exports its interface plus a fixture adapter behind a build flag, labelled in-app as sample data. Fixtures never simulate policy, security, or authorization outcomes as if real.

## 4. Streaming layer

Transport is unknown and is treated as such. The UI depends on an adapter, not a protocol.

```text
UI (useTaskStream)  ->  TaskStreamAdapter  ->  { SSE | chunked-fetch | websocket | poll }
```

- Adapter interface: `start(taskId, { signal }) => AsyncIterable<StreamEvent>`, plus `cancel()`.
- Normalized `StreamEvent` union the UI renders: `token-delta`, `message-complete`, `tool-start`, `tool-update`, `tool-end`, `policy-notice`, `status`, `error`, `done`. Backend event names map into this union inside the adapter only.
- One adapter is selected at build/config time once the backend transport is confirmed. A polling adapter is the documented fallback if no streaming exists, so the UI ships either way.
- `useTaskStream` reduces events into `{ messages, toolEvents, status, error }`, exposes cancel, and reconciles with the persisted task record on `done`. Mid-stream errors surface an inline retry without discarding received content. Cancel aborts the request and calls the backend cancel endpoint if one exists.

## 5. State management

- Server state: TanStack Query only; keys namespaced per domain; invalidate on mutation.
- Stream state: per-task reducer as above.
- UI state: React state/context for panels and theme; filters and search in URL params.
- No global store library. No client persistence of task content, file contents, or audit data beyond the session.

## 6. Component system

Primitives: Button, Input, Textarea, Select, Dialog, Sheet, Tabs, Tooltip, Badge, ScrollArea, Skeleton, EmptyState, ErrorState.

Domain: `MessageList`, `MessageBubble`, `Composer`, `ToolCallCard`, `TaskStatusPill`, `PolicyNotice`, `SourceCitation`, `FileRow`, `ArtifactCard`, `UploadDropzone`, `OcrResultPanel`, `CodeRunPanel`, `ActivityEvent`, `RedactedChip`.

Every async surface ships four states: empty, loading (skeleton), error (with retry), success.

## 7. Security boundaries the frontend respects

- No policy evaluation, allow-listing, path validation, extension/size gatekeeping as a security control, sandbox rules, provider selection rules, or redaction logic in frontend code. Client-side size/type hints exist only to give fast feedback; the backend's verdict is authoritative and always displayed.
- Tool permissions, file access, code execution, and policy outcomes are rendered from backend-returned decisions only — never computed, predicted, or cached to pre-authorize a later action.
- Denied and approval-required states show the backend's user-safe reason text. No internal rule names, config, sandbox images, container flags, provider credentials, or security mechanism details are rendered, logged to the console, or embedded in markup.

## 8. Key flows

**Chat/task**: compose -> optimistic user message -> start task -> stream events -> tool cards and inline policy notices -> terminal state (completed / failed / denied / awaiting approval) with retry or approve where the backend exposes such an action.

**Files**: entries render exactly as the backend classifies them; user uploads and generated artifacts are separated only if the backend provides that distinction, otherwise shown as one neutral list. Uploads use the backend's documented format with per-file progress; server validation errors are shown in plain language. Downloads (including DOCX) go through the backend's download route with its own authorization; the frontend never constructs filesystem paths, never renders absolute host paths, and never generates documents itself.

**Knowledge/RAG**: query -> answer -> retrieved sources listed with whatever the backend returns (snippet, score, origin). A "grounded in knowledge base" marker appears only when the backend indicates knowledge was used.

**Vision/OCR**: upload -> preview -> processing state -> structured result. Extracted content and model interpretation are visually separated only when the backend labels them; otherwise both render under one neutral heading rather than a guessed split.

**Code**: run status (queued / running / succeeded / failed / timed out as the backend reports it), safe stdout/stderr/exit information exactly as returned, readable console block, clear failure state. No sandbox internals, no security logic, no local execution.

**Policy**: inline notices with backend-supplied explanation; approve/decline only if a backend action exists.

**Audit**: reverse-chronological timeline grouped by day, with the filters the backend supports. Renders only fields the backend explicitly exposes; redacted or omitted fields show a neutral chip and are never reconstructed, inferred, or re-requested. Prompts, model outputs, credentials, tokens, sensitive paths, OCR text, raw code, and internals are out of scope for this view by design.

## 9. Design direction

Dark-first, layered near-black surfaces, one restrained accent for state, generous spacing, geometric sans for UI, mono for code and identifiers. Status is color + icon + text. A quiet "Local · Private · Auditable" indicator in the shell. No neon, no dashboard density, no admin-panel sprawl.

## 10. Responsive and accessibility

Desktop-first at 1280px+, usable to 1024; below that sidebars collapse into sheets. Full keyboard paths for send, cancel, panel toggles, and file actions; focus-visible rings; ARIA live regions for streaming and status changes; labelled controls; AA contrast; reduced-motion respected.

## 11. Testing

**Runnable immediately (no backend):**
- Component tests (Vitest + Testing Library): message rendering, tool cards, status pills, policy notices, file rows, OCR panel, code run panel, activity events, all four async states.
- Reducer/adapter tests: the stream reducer against synthetic `StreamEvent` sequences, including mid-stream error, cancel, and out-of-order events.
- Service tests with MSW against fixture responses, plus schema-validation tests proving malformed payloads fail loudly.
- Error-state tests: network failure, timeout, 4xx/5xx normalization, denied/approval states, empty results.
- Accessibility checks on key screens.
- Playwright smoke flows against fixtures: send a task, browse files, run a query, open activity.

**Requires the real backend:**
- Contract tests validating live responses against the agreed schemas.
- Real streaming behavior: transport, event ordering, cancel, reconnect.
- Real upload/download including DOCX round-trip and rejected uploads.
- Real policy denial/approval paths and audit redaction rendering.
- End-to-end task run touching RAG, OCR, and code execution.

## 12. MVP scope

In scope: Agent Workspace, Files with upload/download and DOCX artifacts, Knowledge/RAG, Vision/OCR, Code execution status, inline policy states, Activity, History.

Out of scope for Phase 9: multi-user/roles, theming beyond dark, analytics dashboards, model configuration UI, drag-and-drop file management, offline mode, mobile-optimized layouts, any policy or security configuration surface.

Delivery order: shell + design system -> API client and error model -> Workspace + streaming adapter -> Files -> Knowledge / Vision / Code -> Policy notices + Activity -> accessibility and test hardening.

## 13. Backend contracts required before implementation

For each item: path, method, request schema, response schema, error schema, auth requirement, pagination.

1. Auth: is any required? Scheme, token acquisition/refresh, 401/403 behavior.
2. Tasks: create, get, list, cancel; status enum; message shape.
3. Streaming: transport, event names, delta format, terminal event, mid-stream error behavior, resume support — or confirmation that polling is the only option.
4. Tool events: schema, identifiers, start/update/end/failure payloads.
5. Files: list (entry schema, and whether generated artifacts are flagged), upload (format, field name, limits, progress support), download (URL pattern and auth), delete.
6. Artifacts/DOCX: how generated documents are listed and fetched; content type and filename source.
7. Knowledge: collection list, query request/response, retrieved-source schema, grounding indicator.
8. Vision/OCR: submit endpoint, accepted inputs, sync vs async, result schema, and whether extracted vs interpreted content is labelled.
9. Code: submit/status endpoints, run states, safe output schema, timeout semantics.
10. Policy: decision schema, user-safe reason text, whether an approval action endpoint exists.
11. Audit: list endpoint, supported filters, the explicit list of safe fields exposed, redaction markers, timestamp format/timezone.
12. Cross-cutting: error envelope, rate limits, base URL/CORS for dev, request-id header.

## 14. Assumptions and open questions

- Assumed a single local user with no multi-tenant concerns — confirm.
- Assumed a configurable backend base URL; confirm dev CORS or same-origin serving.
- Assumed tasks persist server-side; if not, History becomes session-only.
- Unknown whether the backend flags generated artifacts, labels OCR extraction vs interpretation, or exposes a grounding indicator. Where it does not, the UI flattens rather than guesses.
- Unknown whether model/router selection is user-visible; a model indicator is added only if it is.
- Unknown whether approval is a UI action or handled out-of-band.
