/**
 * Service layer.
 *
 * Every function below is the typed interface the UI depends on. Each one has
 * a fixture implementation (used until the backend contract is documented) and
 * a clearly marked place where the real `apiRequest` call goes. The real call
 * MUST be written against the documented contract — path, method, request and
 * response schema — and never against a guessed shape.
 *
 * All real calls go through the central apiRequest() boundary (src/lib/http.ts),
 * which normalizes errors and never exposes raw credentials, paths, or tokens.
 */
import { API_BASE_URL, USE_FIXTURES } from "@/lib/config";
import { ApiError, apiRequest } from "@/lib/http";
import {
  fixtureActivity,
  fixtureCodeRuns,
  fixtureCollections,
  fixtureFiles,
  fixtureTask,
  fixtureTasks,
  fixtureVision,
} from "./fixtures";
import type {
  ActivityEventItem,
  CodeRun,
  KnowledgeAnswer,
  KnowledgeCollection,
  KnowledgeSource,
  Task,
  TaskSummary,
  VisionResult,
  WorkspaceEntry,
} from "./types";

// Backend wire payloads, per the FastAPI OpenAPI contract. These reflect the
// API responses exactly; the frontend-facing view models live in `./types`.
interface BackendSearchResultItem {
  chunk_id: string;
  text: string;
  metadata: Record<string, unknown>;
  score: number;
  rank: number;
}

interface BackendSearchResponse {
  query: string;
  total_available: number;
  results: BackendSearchResultItem[];
  count: number;
}

interface BackendCollectionsResponse {
  collections_supported: boolean;
  message: string;
}

interface BackendCodeExecutionResult {
  execution_id: string;
  success: boolean;
  exit_code: number | null;
  stdout: string;
  stderr: string;
  timed_out: boolean;
  duration_seconds: number;
  output_truncated: boolean;
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function notImplemented(what: string): never {
  throw new ApiError(
    `The backend contract for "${what}" has not been provided yet.`,
    0,
    "client",
  );
}

// --- Tasks Service ---

export const tasksService = {
  async list(): Promise<TaskSummary[]> {
    if (USE_FIXTURES) {
      await delay(180);
      return fixtureTasks;
    }
    // Real API call: GET /api/v1/tasks
    const response = await apiRequest<TaskSummary[]>("/api/v1/tasks", {
      method: "GET",
    });
    return response;
  },

  async get(taskId: string): Promise<Task> {
    if (USE_FIXTURES) {
      await delay(200);
      const summary = fixtureTasks.find((t) => t.id === taskId);
      if (!summary) throw new ApiError("This task is not available.", 404, "not_found");
      if (taskId === fixtureTask.id) return fixtureTask;
      return { ...summary, messages: [], toolEvents: [] };
    }
    // Real API call: GET /api/v1/tasks/{taskId}
    const response = await apiRequest<Task>(`/api/v1/tasks/${taskId}`, {
      method: "GET",
    });
    return response;
  },

  async cancel(taskId: string): Promise<TaskSummary> {
    return await apiRequest<TaskSummary>(`/api/v1/tasks/${taskId}/cancel`, {
      method: "POST",
    });
  },
};

// --- Files Service ---

export const filesService = {
  async list(): Promise<WorkspaceEntry[]> {
    if (USE_FIXTURES) {
      await delay(220);
      return fixtureFiles;
    }
    // Real API call: GET /api/v1/files
    const response = await apiRequest<{ files: WorkspaceEntry[] }>("/api/v1/files", {
      method: "GET",
    });
    return response.files;
  },

  /**
   * Uploads go through the backend workspace APIs.
   * The backend performs all validation; client-side checks are only fast feedback, never a gate.
   */
  async upload(files: File[]): Promise<void> {
    if (files.length === 0) return;
    // Real multipart upload against the documented backend contract:
    // POST /api/v1/files/upload with `files` as multipart/form-data fields.
    // The backend validates content and writes into the secure workspace.
    const form = new FormData();
    for (const file of files) {
      form.append("files", file, file.name);
    }
    // No Content-Type is set here: the browser adds the required
    // multipart/form-data boundary for a FormData body. Sending JSON
    // disguised as a file would be rejected by the backend.
    await apiRequest<{ message: string; uploaded: string[] }>("/api/v1/files/upload", {
      method: "POST",
      body: form,
    });
  },

  /**
   * Downloads (including generated DOCX) go through the backend's own
   * authorized download route. The frontend never builds a filesystem path.
   */
  async download(handle: string): Promise<void> {
    if (USE_FIXTURES) {
      await delay(300);
      throw new ApiError(
        "Downloads need the backend download contract before they can be enabled.",
        0,
        "client",
      );
    }
    const response = await fetch(`${API_BASE_URL}/api/v1/files/${handle}`, {
      method: "GET",
    });
    if (!response.ok) {
      const text = await response.text();
      throw new ApiError(
        text || "Failed to download file",
        response.status,
        "server",
      );
    }
    const blob = await response.blob();
    // Create a blob URL and trigger download
    const url = window.URL.createObjectURL(blob);
    // Try to get the filename from the Content-Disposition header
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = handle;
    if (contentDisposition) {
      const match = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
      if (match?.[1]) {
        filename = match[1].replace(/['"]/g, '');
      }
    }
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  },
};

// --- Knowledge Service ---

export const knowledgeService = {
  async collections(): Promise<KnowledgeCollection[]> {
    if (USE_FIXTURES) {
      await delay(200);
      return fixtureCollections;
    }
    // Real API call: GET /api/v1/knowledge/collections
    const response = await apiRequest<BackendCollectionsResponse>("/api/v1/knowledge/collections", {
      method: "GET",
    });
    // The backend reports that collection management is not supported
    // (collections_supported: false). Listing an empty set (rather than
    // fabricating collections) is the honest frontend representation.
    void response;
    return [] as KnowledgeCollection[];
  },

  async query(question: string): Promise<KnowledgeAnswer> {
    if (USE_FIXTURES) {
      await delay(900);
      return {
        id: `k-${Date.now()}`,
        answer:
          "Sample answer. Once the knowledge query contract is available, this text and its sources come straight from the backend.",
        grounded: true,
        sources: [
          {
            id: "ks1",
            title: "Finance — quarterly figures",
            snippet: `Passage matched against: “${question}”`,
            score: 0.81,
            collection: "Finance",
          },
          {
            id: "ks2",
            title: "Internal handbook — reporting",
            snippet: "Reports are compiled at the end of each quarter…",
            score: 0.64,
            collection: "Internal handbook",
          },
        ],
      };
    }
    // Real API call: POST /api/v1/knowledge/search
    const response = await apiRequest<BackendSearchResponse>("/api/v1/knowledge/search", {
      method: "POST",
      body: JSON.stringify({ query: question }),
      headers: { "Content-Type": "application/json" },
    });
    // Map the backend's SearchResponse to the frontend's KnowledgeAnswer.
    // The answer is the concatenated text of the top matching chunks.
    const answer = response.results
      .map((r) => r.text)
      .join(" ")
      .trim() || "No matching passages found in the knowledge base.";
    const sources: KnowledgeSource[] = response.results.map((r) => {
      const metadataTitle = r.metadata["title"];
      const metadataCollection = r.metadata["collection"];
      const source: KnowledgeSource = {
        id: r.chunk_id,
        title: typeof metadataTitle === "string" ? metadataTitle : r.chunk_id,
        snippet: r.text,
        score: r.score,
      };
      if (typeof metadataCollection === "string") source.collection = metadataCollection;
      return source;
    });
    return {
      id: `${response.query}-${Date.now()}`, // Temporary ID based on query and timestamp
      answer,
      grounded: response.results.length > 0,
      sources,
    };
  },
};

// --- Vision Service ---

export const visionService = {
  async process(file: File): Promise<VisionResult> {
    if (USE_FIXTURES) {
      await delay(1400);
      return fixtureVision;
    }
    // Real API call: POST /api/v1/vision/ocr with multipart/form-data
    const form = new FormData();
    form.append("file", file);

    // No Content-Type is set here: the browser adds the required
    // multipart/form-data boundary for a FormData body.
    const response = await apiRequest<{
      image_id: string;
      full_text: string;
      confidence: number;
      page_number: number;
      page_count: number;
      language?: string;
      block_count: number;
      blocks: any[];
      provider: string
    }>("/api/v1/vision/ocr", {
      method: "POST",
      body: form,
    });

    // Map backend OCR response to frontend VisionResult
    return {
      id: response.image_id,
      status: "completed",
      summary: response.full_text,
      // Note: The backend doesn't distinguish between extraction and interpretation
      // so we put the OCR result in summary
    };
  },

  async analyze(file: File, prompt?: string): Promise<VisionResult> {
    if (USE_FIXTURES) {
      await delay(1400);
      return fixtureVision;
    }
    // Real API call: POST /api/v1/vision/analyze
    // The backend expects multipart/form-data for the file and JSON for the prompt
    // We need to create a multipart/form-data request with both parts
    const form = new FormData();
    form.append("file", file);

    // Add prompt as a text field if provided
    if (prompt !== undefined && prompt !== null) {
      form.append("prompt", prompt);
    }

    // No Content-Type is set here: the browser adds the required
    // multipart/form-data boundary for a FormData body.
    const response = await apiRequest<{
      image_id: string;
      description: string;
      confidence: number;
      tags: string[];
      regions: any[];
      provider: string
    }>("/api/v1/vision/analyze", {
      method: "POST",
      body: form,
    });

    // Map backend analyze response to frontend VisionResult
    return {
      id: response.image_id,
      status: "completed",
      interpretation: response.description,
      // Note: We could also put tags or regions in extracted if needed
    };
  }
};

// --- Code Service ---

export const codeService = {
  async list(): Promise<CodeRun[]> {
    if (USE_FIXTURES) {
      await delay(200);
      return fixtureCodeRuns;
    }
    // Real API call: GET /api/v1/code
    const response = await apiRequest<CodeRun[]>("/api/v1/code", {
      method: "GET",
    });
    return response;
  },

  async get(runId: string): Promise<CodeRun> {
    if (USE_FIXTURES) {
      await delay(200);
      const run = fixtureCodeRuns.find((r) => r.id === runId);
      if (!run) throw new ApiError("This run is not available.", 404, "not_found");
      return run;
    }
    // Real API call: GET /api/v1/code/{runId}
    const response = await apiRequest<BackendCodeExecutionResult>(`/api/v1/code/${runId}`, {
      method: "GET",
    });
    // Map the backend's CodeExecutionResult to the frontend's CodeRun.
    const run: CodeRun = {
      id: response.execution_id,
      title: "",
      status: response.success ? "succeeded" : "failed",
      startedAt: "",
      durationMs: Math.round(response.duration_seconds * 1000),
      stdout: response.stdout,
      stderr: response.stderr,
    };
    if (response.exit_code != null) run.exitCode = response.exit_code;
    return run;
  },

  async execute(code: string, language: string = "python", timeout: number = 30.0): Promise<CodeRun> {
    if (USE_FIXTURES) {
      await delay(400);
      return {
        id: `exec-${Date.now()}`,
        title: "Sandbox Run",
        status: "succeeded",
        startedAt: new Date().toISOString(),
        durationMs: 120,
        exitCode: 0,
        stdout: "Execution completed in fixture mode.",
        stderr: "",
      };
    }
    const response = await apiRequest<BackendCodeExecutionResult>("/api/v1/code/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, language, timeout }),
    });
    return {
      id: response.execution_id,
      title: "Sandbox Run",
      status: response.success ? "succeeded" : "failed",
      startedAt: new Date().toISOString(),
      durationMs: Math.round(response.duration_seconds * 1000),
      stdout: response.stdout,
      stderr: response.stderr,
      exitCode: response.exit_code ?? (response.success ? 0 : 1),
    };
  },
};

// --- Activity Service ---

export const activityService = {
  async list(): Promise<ActivityEventItem[]> {
    if (USE_FIXTURES) {
      await delay(240);
      return fixtureActivity;
    }
    // Real API call: GET /api/v1/activity
    // The backend wraps the items in an ActivityResponse ({ items: [...] });
    // the UI route consumes a bare array, so unwrap it here.
    const response = await apiRequest<{ items: ActivityEventItem[] }>("/api/v1/activity", {
      method: "GET",
    });
    return response.items;
  },
};
