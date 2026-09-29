import type { StreamEvent } from "../types";
import type { StreamStartOptions, TaskStreamAdapter } from "./adapter";

/**
 * Fixture adapter — emits a realistic normalized event sequence so the
 * streaming UI can be built and tested before the backend transport is known.
 * It performs no policy or security logic; it only replays sample events.
 */
const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });

export const fixtureStreamAdapter: TaskStreamAdapter = {
  name: "fixture",
  async *start(taskId: string, options: StreamStartOptions): AsyncGenerator<StreamEvent> {
    const { signal } = options;
    const messageId = `${taskId}-agent`;

    yield { type: "status", status: "running" };
    await sleep(300, signal);

    yield {
      type: "tool-start",
      tool: { id: "tool-plan", label: "Planning the task", status: "running" },
    };
    await sleep(500, signal);
    yield {
      type: "tool-end",
      tool: { id: "tool-plan", label: "Planning the task", status: "succeeded" },
    };

    yield {
      type: "tool-start",
      tool: { id: "tool-kb", label: "Knowledge base lookup", status: "running" },
    };
    await sleep(700, signal);
    yield {
      type: "tool-end",
      tool: {
        id: "tool-kb",
        label: "Knowledge base lookup",
        status: "succeeded",
        detail: "2 passages",
      },
    };

    const reply =
      "Here is a sample streamed response. Once the backend streaming contract is confirmed, this text arrives from the local model gateway and the tool steps above reflect real tool activity.";

    for (const word of reply.split(" ")) {
      await sleep(35, signal);
      yield { type: "token-delta", messageId, text: `${word} ` };
    }

    await sleep(200, signal);
    yield { type: "status", status: "completed" };
    yield { type: "done" };
  },
};
