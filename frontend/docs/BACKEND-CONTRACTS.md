# Backend contracts required before the frontend can connect

The frontend is built against typed service interfaces (`src/services/api.ts`)
and a transport-agnostic stream adapter (`src/services/stream/`). Nothing here
is guessed: until each contract below is supplied, the matching service throws
a clear "contract not provided" error and the UI runs on labelled sample data
(`VITE_USE_FIXTURES`).

For every endpoint we need: path, HTTP method, request schema, response schema,
error schema, auth requirement, and pagination behaviour.

1. **Auth** — is any required? Scheme, token acquisition/refresh, 401/403 behaviour.
2. **Tasks** — create, get, list, cancel; status enum; message shape.
3. **Streaming** — transport (SSE / chunked / websocket), event names, delta
   format, terminal event, mid-stream error behaviour, resume support — or
   confirmation that polling is the only option.
4. **Tool events** — schema, identifiers, start/update/end/failure payloads.
5. **Files** — list (entry schema; is a generated artifact flagged?), upload
   (format, field name, limits, progress), download (URL pattern + auth), delete.
6. **Artifacts / DOCX** — how generated documents are listed and fetched;
   content type and filename source.
7. **Knowledge** — collection list, document listing, query request/response,
   retrieved-source schema, grounding indicator.
8. **Vision / OCR** — submit endpoint, accepted inputs, sync vs async, result
   schema, and whether extracted content and model interpretation are labelled
   separately.
9. **Code** — submit/status endpoints, run states, safe output schema, timeout
   semantics.
10. **Policy** — decision schema, user-safe reason text, whether an approval
    action endpoint exists.
11. **Audit** — list endpoint, supported filters, the explicit list of safe
    fields exposed, redaction markers, timestamp format/timezone.
12. **Cross-cutting** — error envelope, rate limits, base URL / CORS for dev,
    request-id header.

## Frontend security posture

The backend is the sole authority. The frontend does not implement or duplicate
the policy engine, workspace security, sandbox restrictions, provider security,
or audit redaction. It does not construct filesystem paths, does not predict or
cache authorization outcomes, and renders only the safe fields the backend
explicitly exposes.

## Where the real calls go

- `src/lib/http.ts` — the single HTTP boundary and normalized error model.
- `src/services/api.ts` — one function per operation; replace each fixture
  branch with the documented `apiRequest` call plus response validation.
- `src/services/stream/` — add the confirmed transport adapter and select it in
  `index.ts`. No component imports a concrete adapter.

## Tests

- `bunx vitest run` — component, reducer, and error-state tests that run today.
- Contract, streaming, upload/download, policy, and audit-redaction tests
  require the real backend and are written once the contracts above land.
