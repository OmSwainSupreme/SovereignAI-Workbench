import { API_BASE_URL } from "./config";

/**
 * Single HTTP boundary for every backend call.
 *
 * No component or feature module calls fetch directly. Error shapes are
 * normalized here so the UI has one error model. The backend remains the sole
 * authority for authorization: a 401/403 is surfaced as-is, never worked
 * around, retried with different parameters, or cached as a local decision.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly kind: "network" | "unauthorized" | "forbidden" | "not_found" | "server" | "client";
  readonly requestId?: string | undefined;

  constructor(
    message: string,
    status: number,
    kind: ApiError["kind"],
    requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;
    this.requestId = requestId;
  }
}

function kindFor(status: number): ApiError["kind"] {
  if (status === 401) return "unauthorized";
  if (status === 403) return "forbidden";
  if (status === 404) return "not_found";
  if (status >= 500) return "server";
  return "client";
}

export interface RequestOptions extends RequestInit {
  /** Parsed response validator, supplied once real contracts are known. */
  parse?: (data: unknown) => unknown;
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { parse, ...init } = options;
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init.headers ?? {}) },
    });
  } catch (cause) {
    throw new ApiError(
      cause instanceof DOMException && cause.name === "AbortError"
        ? "Request cancelled."
        : "Could not reach the local backend.",
      0,
      "network",
    );
  }

  const requestId = response.headers.get("x-request-id") ?? undefined;

  if (!response.ok) {
    // The backend owns user-facing error text; we do not invent explanations.
    // FastAPI surfaces HTTPException details as `detail`; app endpoints may
    // use `message` or `error`. Read all three so the real reason reaches the UI.
    let message = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as {
        message?: string;
        error?: string;
        detail?: unknown;
      };
      message =
        body.message ??
        body.error ??
        (typeof body.detail === "string" ? body.detail : undefined) ??
        message;
    } catch {
      /* non-JSON error body: keep the generic message */
    }
    throw new ApiError(message, response.status, kindFor(response.status), requestId);
  }

  const data: unknown = response.status === 204 ? null : await response.json();
  return (parse ? parse(data) : data) as T;
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.kind === "network") return "Could not reach the local backend.";
    if (error.kind === "unauthorized" || error.kind === "forbidden")
      return error.message || "This action isn't available to you.";
    return error.message;
  }
  if (error instanceof Error && error.message) return error.message;
  return "Something went wrong.";
}
