import { describe, expect, it } from "vitest";
import { initialStreamState, streamReducer, type StreamState } from "./streamReducer";
import type { StreamEvent } from "@/services/types";

const run = (events: StreamEvent[], from: StreamState = initialStreamState) =>
  events.reduce((state, event) => streamReducer(state, { type: "event", event }), from);

describe("streamReducer", () => {
  it("accumulates token deltas into a single pending message", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "token-delta", messageId: "m1", text: "Hello " },
      { type: "token-delta", messageId: "m1", text: "world" },
    ]);

    expect(state.messages).toHaveLength(1);
    expect(state.messages[0]?.content).toBe("Hello world");
    expect(state.messages[0]?.pending).toBe(true);
    expect(state.status).toBe("running");
  });

  it("clears pending state and completes on done", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "token-delta", messageId: "m1", text: "hi" },
      { type: "done" },
    ]);

    expect(state.messages[0]?.pending).toBe(false);
    expect(state.status).toBe("completed");
  });

  it("upserts tool events by id rather than duplicating them", () => {
    const state = run([
      { type: "tool-start", tool: { id: "t1", label: "Read file", status: "running" } },
      {
        type: "tool-end",
        tool: { id: "t1", label: "Read file", status: "succeeded", detail: "report.pdf" },
      },
    ]);

    expect(state.toolEvents).toHaveLength(1);
    expect(state.toolEvents[0]?.status).toBe("succeeded");
    expect(state.toolEvents[0]?.detail).toBe("report.pdf");
  });

  it("keeps received content when the stream errors mid-way", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "token-delta", messageId: "m1", text: "partial" },
      { type: "error", message: "Connection lost." },
    ]);

    expect(state.status).toBe("failed");
    expect(state.error).toBe("Connection lost.");
    expect(state.messages[0]?.content).toBe("partial");
    expect(state.messages[0]?.pending).toBe(false);
  });

  it("records a backend policy denial without inventing a decision", () => {
    const state = run([
      { type: "status", status: "running" },
      {
        type: "policy-notice",
        notice: { id: "p1", decision: "denied", reason: "This action is not allowed." },
      },
    ]);

    expect(state.status).toBe("denied");
    expect(state.policy?.reason).toBe("This action is not allowed.");
  });

  it("marks running tools as stopped when the user cancels", () => {
    const running = run([
      { type: "status", status: "running" },
      { type: "tool-start", tool: { id: "t1", label: "Working", status: "running" } },
    ]);
    const state = streamReducer(running, { type: "cancelled" });

    expect(state.status).toBe("cancelled");
    expect(state.toolEvents[0]?.status).toBe("failed");
  });

  it("tracks phase and model updates in real-time", () => {
    const state = run([
      { type: "status", status: "running" },
      {
        type: "phase",
        phase: "generating",
        label: "Generating response with qwen2.5-coder:3b...",
        model: "qwen2.5-coder:3b",
        elapsedSeconds: 8.5,
      },
    ]);

    expect(state.phase).toBe("generating");
    expect(state.phaseLabel).toBe("Generating response with qwen2.5-coder:3b...");
    expect(state.model).toBe("qwen2.5-coder:3b");
    expect(state.elapsedSeconds).toBe(8.5);
  });

  it("records task-id for cancellation control", () => {
    const state = run([
      { type: "task-id", id: "task-abc-123" },
    ]);

    expect(state.currentTaskId).toBe("task-abc-123");
  });
  it("fails task if done event arrives with only thinking and no assistant content", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "thinking-delta", messageId: "m1", text: "Reasoning only..." },
      { type: "done" },
    ]);

    expect(state.status).toBe("failed");
    expect(state.phase).toBe("failed");
    expect(state.phaseLabel).toBe("Generation interrupted: No final answer produced");
    expect(state.messages[0]?.pending).toBe(false);
  });

  it("fails task if done event arrives with empty content", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "token-delta", messageId: "m1", text: "   " },
      { type: "done" },
    ]);

    expect(state.status).toBe("failed");
    expect(state.phase).toBe("failed");
    expect(state.phaseLabel).toBe("Generation interrupted: No final answer produced");
  });

  it("completes task if done event arrives with valid assistant content", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "thinking-delta", messageId: "m1", text: "Reasoning..." },
      { type: "token-delta", messageId: "m1", text: "Valid final answer." },
      { type: "done" },
    ]);

    expect(state.status).toBe("completed");
    expect(state.phase).toBe("completed");
    expect(state.phaseLabel).toBe("Completed");
    expect(state.messages[0]?.content).toBe("Valid final answer.");
    expect(state.messages[0]?.pending).toBe(false);
  });
});
