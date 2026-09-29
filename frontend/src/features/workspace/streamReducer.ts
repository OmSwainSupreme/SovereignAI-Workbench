import type {
  ChatMessage,
  PolicyNotice,
  StreamEvent,
  TaskStatus,
  ToolEvent,
} from "@/services/types";

export interface StreamState {
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  status: TaskStatus;
  policy: PolicyNotice | null;
  error: string | null;
  phase: string | null;
  phaseLabel: string | null;
  model: string | null;
  elapsedSeconds: number | null;
  currentTaskId: string | null;
}

export const initialStreamState: StreamState = {
  messages: [],
  toolEvents: [],
  status: "idle",
  policy: null,
  error: null,
  phase: null,
  phaseLabel: null,
  model: null,
  elapsedSeconds: null,
  currentTaskId: null,
};

export type StreamAction =
  | { type: "reset"; messages?: ChatMessage[]; toolEvents?: ToolEvent[]; status?: TaskStatus }
  | { type: "append-message"; message: ChatMessage }
  | { type: "event"; event: StreamEvent }
  | { type: "cancelled" };

function upsertTool(tools: ToolEvent[], tool: ToolEvent): ToolEvent[] {
  const index = tools.findIndex((t) => t.id === tool.id);
  if (index === -1) return [...tools, tool];
  const next = [...tools];
  next[index] = { ...next[index], ...tool };
  return next;
}

function applyThinkingDelta(messages: ChatMessage[], messageId: string, text: string): ChatMessage[] {
  const index = messages.findIndex((m) => m.id === messageId);
  if (index === -1) {
    return [
      ...messages,
      {
        id: messageId,
        role: "agent",
        content: "",
        thinking: text,
        createdAt: new Date().toISOString(),
        pending: true,
      },
    ];
  }
  const next = [...messages];
  const current = next[index]!;
  next[index] = { ...current, thinking: (current.thinking || "") + text, pending: true };
  return next;
}

function applyDelta(messages: ChatMessage[], messageId: string, text: string): ChatMessage[] {
  const index = messages.findIndex((m) => m.id === messageId);
  if (index === -1) {
    return [
      ...messages,
      {
        id: messageId,
        role: "agent",
        content: text,
        createdAt: new Date().toISOString(),
        pending: true,
      },
    ];
  }
  const next = [...messages];
  const current = next[index]!;
  next[index] = { ...current, content: current.content + text, pending: true };
  return next;
}

export function streamReducer(state: StreamState, action: StreamAction): StreamState {
  switch (action.type) {
    case "reset":
      return {
        ...initialStreamState,
        messages: action.messages ?? [],
        toolEvents: action.toolEvents ?? [],
        status: action.status ?? "idle",
      };

    case "append-message":
      return { ...state, messages: [...state.messages, action.message] };

    case "cancelled":
      return {
        ...state,
        status: "cancelled",
        phase: "cancelled",
        phaseLabel: "Cancelled by user",
        messages: state.messages.map((m) => ({ ...m, pending: false })),
        toolEvents: state.toolEvents.map((t) =>
          t.status === "running" ? { ...t, status: "failed" } : t,
        ),
      };

    case "event": {
      const event = action.event;
      switch (event.type) {
        case "status":
          return { ...state, status: event.status };
        case "task-id":
          return { ...state, currentTaskId: event.id };
        case "phase":
          return {
            ...state,
            phase: event.phase,
            phaseLabel: event.label,
            model: event.model ?? state.model,
            elapsedSeconds: event.elapsedSeconds ?? state.elapsedSeconds,
          };
        case "token-delta":
          return {
            ...state,
            messages: applyDelta(state.messages, event.messageId, event.text),
          };
        case "thinking-delta":
          return {
            ...state,
            messages: applyThinkingDelta(state.messages, event.messageId, event.text),
          };
        case "message-complete": {
          const index = state.messages.findIndex((m) => m.id === event.message.id);
          const messages = [...state.messages];
          if (index === -1) messages.push({ ...event.message, pending: false });
          else messages[index] = { ...event.message, pending: false };
          return { ...state, messages };
        }
        case "tool-start":
        case "tool-update":
        case "tool-end":
          return { ...state, toolEvents: upsertTool(state.toolEvents, event.tool) };
        case "policy-notice":
          return {
            ...state,
            policy: event.notice,
            status: event.notice.decision === "denied" ? "denied" : state.status,
          };
        case "error":
          return {
            ...state,
            error: event.message,
            status: "failed",
            phase: "failed",
            phaseLabel: "Task failed",
            messages: state.messages.map((m) => ({ ...m, pending: false })),
          };
        case "done": {
          const hasAssistantContent = state.messages.some(
            (m) =>
              (m.role === "agent" || m.role === "assistant") &&
              Boolean(m.content && m.content.trim().length > 0),
          );
          if (!hasAssistantContent) {
            return {
              ...state,
              status: "failed",
              phase: "failed",
              phaseLabel: "Generation interrupted: No final answer produced",
              error: "Generation interrupted: No final answer produced.",
              messages: state.messages.map((m) => ({ ...m, pending: false })),
            };
          }
          return {
            ...state,
            phase: "completed",
            phaseLabel: "Completed",
            messages: state.messages.map((m) => ({ ...m, pending: false })),
            status:
              state.status === "running" || state.status === "queued"
                ? "completed"
                : state.status,
          };
        }
        default:
          return state;
      }
    }

    default:
      return state;
  }
}
