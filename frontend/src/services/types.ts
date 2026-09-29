/**
 * Domain types for the SovereignAI Workbench frontend.
 *
 * IMPORTANT: these are FRONTEND-FACING view models, not backend contracts.
 * The real backend schemas are not yet documented (see the Phase 9 plan,
 * "Backend contracts required"). Each service adapter is responsible for
 * mapping real backend payloads into these shapes once the contracts exist.
 */

export type TaskStatus =
  | "idle"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "denied"
  | "awaiting_approval";

export type PolicyDecision = "allowed" | "denied" | "requires_approval";

export interface PolicyNotice {
  id: string;
  decision: PolicyDecision;
  /** User-safe explanation supplied by the backend. Never generated locally. */
  reason: string;
  /** Present only when the backend exposes an approval action. */
  approvable?: boolean;
}

export type ToolStatus = "running" | "succeeded" | "failed";

export interface ToolEvent {
  id: string;
  /** Display label supplied by the backend. */
  label: string;
  status: ToolStatus;
  /** Short, non-sensitive summary supplied by the backend. */
  detail?: string;
}

export interface KnowledgeSource {
  id: string;
  title: string;
  snippet: string;
  score?: number;
  collection?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "agent";
  content: string;
  createdAt: string;
  /** Attachment names only — no local file contents are persisted. */
  attachments?: string[];
  /** Present only when the backend reports knowledge-base grounding. */
  sources?: KnowledgeSource[];
  grounded?: boolean;
  pending?: boolean;
  thinking?: string;
}

export interface TaskSummary {
  id: string;
  title: string;
  status: TaskStatus;
  updatedAt: string;
  preview?: string;
  phase?: string;
  phaseLabel?: string;
  model?: string;
  elapsedSeconds?: number;
}

export interface Task extends TaskSummary {
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  policy?: PolicyNotice;
}

/** Backend-classified origin. `unknown` = the backend did not tell us. */
export type FileOrigin = "user_upload" | "generated" | "unknown";

export interface WorkspaceEntry {
  id: string;
  name: string;
  kind: "file" | "folder";
  origin: FileOrigin;
  sizeBytes?: number;
  updatedAt: string;
  contentType?: string;
  /** Opaque handle echoed back to the backend. Never a real filesystem path. */
  handle: string;
  downloadable: boolean;
}

export interface KnowledgeCollection {
  id: string;
  name: string;
  documentCount: number;
  updatedAt: string;
}

export interface KnowledgeAnswer {
  id: string;
  answer: string;
  grounded: boolean;
  sources: KnowledgeSource[];
}

export type VisionStatus = "queued" | "processing" | "completed" | "failed";

export interface VisionResult {
  id: string;
  status: VisionStatus;
  /** Only rendered when the backend labels extraction separately. */
  extracted?: { label: string; fields: { key: string; value: string }[] };
  /** Only rendered when the backend labels interpretation separately. */
  interpretation?: string;
  /** Used when the backend does not distinguish the two. */
  summary?: string;
  error?: string;
}

export type CodeRunStatus = "queued" | "running" | "succeeded" | "failed" | "timed_out";

export interface CodeRun {
  id: string;
  title: string;
  status: CodeRunStatus;
  startedAt: string;
  durationMs?: number;
  exitCode?: number;
  stdout?: string;
  stderr?: string;
  policy?: PolicyNotice;
}

export type ActivityStatus = "ok" | "denied" | "failed" | "pending";

export interface ActivityEventItem {
  id: string;
  timestamp: string;
  action: string;
  status: ActivityStatus;
  /** Safe, backend-exposed detail only. */
  detail?: string;
  /** Field names the backend reports as redacted. Values are never requested. */
  redactedFields?: string[];
}

/** Normalized stream events. Adapters map backend events into this union. */
export type StreamEvent =
  | { type: "status"; status: TaskStatus }
  | { type: "task-id"; id: string }
  | { type: "phase"; phase: string; label: string; model?: string; elapsedSeconds?: number }
  | { type: "token-delta"; messageId: string; text: string }
  | { type: "thinking-delta"; messageId: string; text: string }
  | { type: "message-complete"; message: ChatMessage }
  | { type: "tool-start"; tool: ToolEvent }
  | { type: "tool-update"; tool: ToolEvent }
  | { type: "tool-end"; tool: ToolEvent }
  | { type: "policy-notice"; notice: PolicyNotice }
  | { type: "error"; message: string }
  | { type: "done" };
