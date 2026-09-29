import { tasksService } from "@/services/api";
import { useCallback, useEffect, useReducer, useRef } from "react";
import { getTaskStreamAdapter } from "@/services/stream";
import type { ChatMessage, ToolEvent } from "@/services/types";
import { initialStreamState, streamReducer } from "./streamReducer";

interface UseTaskStreamOptions {
  taskId: string;
  initialMessages?: ChatMessage[];
  initialToolEvents?: ToolEvent[];
}

export function useTaskStream({
  taskId,
  initialMessages,
  initialToolEvents,
}: UseTaskStreamOptions) {
  const [state, dispatch] = useReducer(streamReducer, initialStreamState);
  const controllerRef = useRef<AbortController | null>(null);
  const lastPromptRef = useRef<string>("");

  useEffect(() => {
    dispatch({
      type: "reset",
      messages: initialMessages ?? [],
      toolEvents: initialToolEvents ?? [],
      status: (initialMessages?.length ?? 0) > 0 ? "completed" : "idle",
    });
    return () => controllerRef.current?.abort();
  }, [taskId, initialMessages, initialToolEvents]);

  const run = useCallback(
    async (prompt: string, attachments: string[] = []) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      lastPromptRef.current = prompt;

      dispatch({
        type: "append-message",
        message: {
          id: `u-${Date.now()}`,
          role: "user",
          content: prompt,
          createdAt: new Date().toISOString(),
          ...(attachments.length > 0 ? { attachments } : {}),
        },
      });

      try {
        const adapter = getTaskStreamAdapter();
        for await (const event of adapter.start(taskId, {
          signal: controller.signal,
          prompt,
          ...(attachments.length > 0 ? { attachments } : {}),
        })) {
          if (controller.signal.aborted) break;
          dispatch({ type: "event", event });
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        dispatch({
          type: "event",
          event: {
            type: "error",
            message:
              error instanceof Error ? error.message : "The task stream stopped unexpectedly.",
          },
        });
      }
    },
    [taskId],
  );

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    const activeId = state.currentTaskId || (taskId !== "new" ? taskId : null);
    if (activeId) {
      void tasksService.cancel(activeId).catch(() => {});
    }
    dispatch({ type: "cancelled" });
  }, [state.currentTaskId, taskId]);

  const retry = useCallback(() => {
    if (lastPromptRef.current) void run(lastPromptRef.current);
  }, [run]);

  return { ...state, run, cancel, retry, isStreaming: state.status === "running" };
}
