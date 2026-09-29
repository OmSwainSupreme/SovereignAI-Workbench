import { apiRequest } from "@/lib/http";
import type { StreamEvent, Task } from "../types";
import type { StreamStartOptions, TaskStreamAdapter } from "./adapter";

/**
 * Polling task stream adapter.
 *
 * The backend exposes a submit-then-poll contract (POST /api/v1/tasks followed
 * by GET /api/v1/tasks/{id}); there is no server-push streaming transport. This
 * adapter drives that contract and maps each poll into the normalized
 * `StreamEvent` union exactly like a stream would, so the UI never needs to
 * know the transport.
 *
 * Task states come back as snapshots: a task stays "queued" while the agent
 * works and flips to "completed"/"failed" with the full message and tool-event
 * list once it finishes. Tool events and agent messages are emitted once each,
 * so re-polling a finished task is idempotent in the reducer.
 */

const INITIAL_POLL_INTERVAL_MS = 250;
const ACTIVE_POLL_INTERVAL_MS = 800;
const POLL_INTERVAL_MS = 1000;

function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    if (signal.aborted) {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

interface SubmittedTask {
  id: string;
}

export const pollingStreamAdapter: TaskStreamAdapter = {
  name: "polling",
  async *start(_taskId: string, options: StreamStartOptions): AsyncGenerator<StreamEvent> {
    const { signal, prompt, attachments } = options;

    yield { type: "status", status: "queued" };

    // Submit the task. The backend runs it in the background and returns the ID
    // immediately; on failure this surfaces the real error to the UI.
    // Attached file names ride along in the payload so the backend can
    // validate them and hand the agent the correct workspace-relative paths.
    const submitted = await apiRequest<SubmittedTask>("/api/v1/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: prompt,
        ...(attachments?.length ? { attachments } : {}),
      }),
      signal,
    });
    const taskId = submitted.id;
    yield { type: "task-id", id: taskId };
    // The backend keeps the task at "queued" while the agent works; the UI has
    // already moved to "running" so don't let the first poll drag it back.
    yield { type: "status", status: "running" };

    const seenTools = new Set<string>();
    const completedTools = new Set<string>();
    const seenMessages = new Set<string>();
    const lastContentByMsg = new Map<string, string>();
    const lastThinkingByMsg = new Map<string, string>();
    let lastStatus = "running";
    let lastPhase = "";
    let lastPhaseLabel = "";
    let pollIteration = 0;
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 5;

    while (true) {
      if (signal.aborted) return;

      let task: Task;
      try {
        task = await apiRequest<Task>(`/api/v1/tasks/${taskId}`, {
          method: "GET",
          signal,
        });
        consecutiveErrors = 0;
      } catch (err: any) {
        if (signal.aborted || err?.name === "AbortError") return;
        consecutiveErrors++;
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          throw err;
        }
        await abortableSleep(Math.min(1000 * consecutiveErrors, 4000), signal);
        continue;
      }

      if (task.status !== lastStatus) {
        lastStatus = task.status;
        yield { type: "status", status: task.status };
      }

      if (task.phase && (task.phase !== lastPhase || task.phaseLabel !== lastPhaseLabel)) {
        lastPhase = task.phase;
        lastPhaseLabel = task.phaseLabel || "";
        yield {
          type: "phase",
          phase: task.phase,
          label: task.phaseLabel || task.phase,
          model: task.model,
          elapsedSeconds: task.elapsedSeconds,
        };
      }

      for (const tool of task.toolEvents) {
        if (!seenTools.has(tool.id)) {
          seenTools.add(tool.id);
          yield { type: "tool-start", tool };
        }
        if (tool.status !== "running" && !completedTools.has(tool.id)) {
          completedTools.add(tool.id);
          yield { type: "tool-end", tool };
        }
      }

      for (const message of task.messages) {
        if (message.role !== "agent") continue;
        const prevText = lastContentByMsg.get(message.id) ?? "";
        const prevThinking = lastThinkingByMsg.get(message.id) ?? "";
        if (message.pending) {
          if (message.thinking && message.thinking.length > prevThinking.length) {
            const delta = message.thinking.slice(prevThinking.length);
            lastThinkingByMsg.set(message.id, message.thinking);
            yield { type: "thinking-delta", messageId: message.id, text: delta };
          }
          if (message.content && message.content.length > prevText.length) {
            const delta = message.content.slice(prevText.length);
            lastContentByMsg.set(message.id, message.content);
            yield { type: "token-delta", messageId: message.id, text: delta };
          }
        } else {
          if (!seenMessages.has(message.id)) {
            seenMessages.add(message.id);
            yield { type: "message-complete", message };
          }
        }
      }

      if (task.status === "completed") {
        yield { type: "done" };
        return;
      }
      if (task.status === "cancelled") {
        return;
      }
      if (task.status === "failed") {
        yield { type: "error", message: task.phaseLabel || "The task failed to complete." };
        return;
      }

      pollIteration++;
      const currentInterval =
        pollIteration <= 2
          ? INITIAL_POLL_INTERVAL_MS
          : task.status === "running"
            ? ACTIVE_POLL_INTERVAL_MS
            : POLL_INTERVAL_MS;
      await abortableSleep(currentInterval, signal);
    }
  },
};
