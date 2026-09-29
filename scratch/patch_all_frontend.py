import re
from pathlib import Path

# 1. Update C:\lovable\src\services\api.ts - add cancel method to tasksService
api_path = Path(r"C:\lovable\src\services\api.ts")
api_text = api_path.read_text(encoding="utf-8")
if "async cancel(" not in api_text:
    old_tasks_service = """    // Real API call: GET /api/v1/tasks/{taskId}
    const response = await apiRequest<Task>(`/api/v1/tasks/${taskId}`, {
      method: "GET",
    });
    return response;
  },
};"""
    new_tasks_service = """    // Real API call: GET /api/v1/tasks/{taskId}
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
};"""
    if old_tasks_service in api_text:
        api_text = api_text.replace(old_tasks_service, new_tasks_service)
        api_path.write_text(api_text, encoding="utf-8")
        print("Updated api.ts with tasksService.cancel")
    else:
        print("Failed to find old_tasks_service block in api.ts")
else:
    print("api.ts already has cancel method")

# 2. Update C:\lovable\src\features\workspace\streamReducer.ts - fix default reset status
reducer_path = Path(r"C:\lovable\src\features\workspace\streamReducer.ts")
reducer_text = reducer_path.read_text(encoding="utf-8")
old_reset = """    case "reset":
      return {
        ...initialStreamState,
        messages: action.messages ?? [],
        toolEvents: action.toolEvents ?? [],
        status: action.status ?? "queued",
      };"""
new_reset = """    case "reset":
      return {
        ...initialStreamState,
        messages: action.messages ?? [],
        toolEvents: action.toolEvents ?? [],
        status: action.status ?? "idle",
      };"""
if old_reset in reducer_text:
    reducer_text = reducer_text.replace(old_reset, new_reset)
    reducer_path.write_text(reducer_text, encoding="utf-8")
    print("Updated streamReducer.ts default reset status to 'idle'")
else:
    print("streamReducer.ts already updated or pattern mismatch")

# 3. Update C:\lovable\src\services\stream\pollingAdapter.ts - token delta streaming
polling_path = Path(r"C:\lovable\src\services\stream\pollingAdapter.ts")
polling_text = polling_path.read_text(encoding="utf-8")
old_msg_loop = """      for (const message of task.messages) {
        if (message.role !== "agent" || seenMessages.has(message.id)) continue;
        seenMessages.add(message.id);
        yield { type: "message-complete", message };
      }"""
new_msg_loop = """      for (const message of task.messages) {
        if (message.role !== "agent") continue;
        const prevText = lastContentByMsg.get(message.id) ?? "";
        if (message.pending) {
          if (message.content.length > prevText.length) {
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
      }"""
if old_msg_loop in polling_text:
    polling_text = polling_text.replace(
        "    const seenMessages = new Set<string>();",
        "    const seenMessages = new Set<string>();\n    const lastContentByMsg = new Map<string, string>();"
    )
    polling_text = polling_text.replace(old_msg_loop, new_msg_loop)
    polling_path.write_text(polling_text, encoding="utf-8")
    print("Updated pollingAdapter.ts with real token-delta streaming")
else:
    print("pollingAdapter.ts message loop already updated or pattern mismatch")

# 4. Update C:\lovable\src\features\workspace\TaskView.tsx
taskview_path = Path(r"C:\lovable\src\features\workspace\TaskView.tsx")
tv_text = taskview_path.read_text(encoding="utf-8")

# 4a. Remove redundant green status message above input
redundant_green = """          {isBusy ? (
            <div className="mb-2 flex items-center justify-between rounded-md border border-primary/25 bg-primary/10 px-3 py-2 text-xs text-primary font-medium">
              <div className="flex items-center gap-2">
                <Loader2 className="size-3.5 animate-spin" />
                <span>AI is currently working on your task</span>
              </div>
              <span className="text-mono text-[11px] text-muted-foreground">
                {stream.phaseLabel || "Processing..."}
              </span>
            </div>
          ) : null}"""
if redundant_green in tv_text:
    tv_text = tv_text.replace(redundant_green, "")
    print("Removed redundant green input message from TaskView.tsx")

# 4b. Add accept attribute to file input
old_input = """            <input
              ref={fileRef}
              type="file"
              multiple
              disabled={isBusy}
              className="sr-only"
              onChange={(event) => {"""
new_input = """            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,.py,.json,.yaml,.csv,.pdf,.png,.jpg,.jpeg,.webp"
              multiple
              disabled={isBusy}
              className="sr-only"
              onChange={(event) => {"""
if old_input in tv_text:
    tv_text = tv_text.replace(old_input, new_input)
    print("Added accept attribute to file input in TaskView.tsx")

# 4c. Redesign AgentProgressCard with collapsible thinking
card_pattern = re.compile(r"function AgentProgressCard\(.*?\n\)\s*\{.*?\n\}\n\nexport function TaskView", re.DOTALL)
new_card_code = """function AgentProgressCard({
  status,
  phase,
  phaseLabel,
  model,
  elapsedSeconds,
  toolEvents,
  onCancel,
}: {
  status: TaskStatus;
  phase: string | null;
  phaseLabel: string | null;
  model: string | null;
  elapsedSeconds: number | null;
  toolEvents?: ToolEvent[];
  onCancel: () => void;
}) {
  const [elapsed, setElapsed] = useState(elapsedSeconds ? Math.round(elapsedSeconds) : 0);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (status !== "running" && status !== "queued") return;
    const start = Date.now() - (elapsedSeconds ? elapsedSeconds * 1000 : 0);
    const interval = setInterval(() => {
      setElapsed(Math.max(1, Math.round((Date.now() - start) / 1000)));
    }, 1000);
    return () => clearInterval(interval);
  }, [status, elapsedSeconds]);

  if (status !== "running" && status !== "queued") return null;

  const stages = [
    { key: "understanding", label: "Understanding request", active: phase === "understanding" },
    { key: "planning", label: "Planning steps", active: phase === "planning" },
    { key: "routing", label: model ? `Model selected: ${model}` : "Selecting model", active: phase === "routing" },
    ...(toolEvents && toolEvents.length > 0
      ? toolEvents.map((t) => ({
          key: t.id,
          label: `${t.label} (${t.status})`,
          active: t.status === "running",
          done: t.status === "succeeded",
        }))
      : [{ key: "tool_execution", label: "Executing tools", active: phase === "tool_execution" || phase === "tool_request" || phase === "observing" }]
    ),
    { key: "generating", label: "Generating response", active: phase === "generating" },
    { key: "verifying", label: "Verifying output", active: phase === "verifying" },
  ];

  const phaseOrder = ["understanding", "planning", "routing", "tool_execution", "generating", "verifying", "completed"];
  const currentIdx = phase ? phaseOrder.indexOf(phase) : (status === "queued" ? -1 : 0);

  return (
    <div className="rounded-xl border border-primary/30 bg-surface/90 p-3.5 shadow-md backdrop-blur">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className="relative flex size-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
            <span className="relative inline-flex size-2.5 rounded-full bg-primary" />
          </span>
          <span className="text-sm font-medium text-foreground">
            {phaseLabel || (status === "queued" ? "Queued in task runner..." : "Agent working on request...")}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {model ? (
            <span className="rounded bg-primary/10 px-2 py-0.5 text-mono text-xs font-semibold text-primary">
              {model}
            </span>
          ) : null}
          <span className="text-mono text-xs text-muted-foreground font-medium">
            {elapsed}s
          </span>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onCancel}
            className="h-7 px-2 text-xs text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
          >
            Cancel
          </Button>
        </div>
      </div>

      <div className="mt-2.5 pt-2 border-t border-border/40 flex flex-col gap-2">
        <button
          type="button"
          onClick={() => setIsExpanded((prev) => !prev)}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground font-medium transition-colors w-fit"
          aria-expanded={isExpanded}
        >
          <span>Thinking</span>
          <span className="text-[10px]">{isExpanded ? "▴" : "▾"}</span>
        </button>

        {isExpanded ? (
          <div className="pl-2 border-l-2 border-primary/30 space-y-1.5 py-1 text-xs">
            {stages.map((st, idx) => {
              const isDone = st.done ?? (currentIdx > idx);
              const isCurrent = st.active ?? (currentIdx === idx);
              return (
                <div
                  key={st.key}
                  className={cn(
                    "flex items-center gap-2 transition-colors",
                    isCurrent
                      ? "text-primary font-semibold"
                      : isDone
                        ? "text-foreground/90 font-medium"
                        : "text-muted-foreground/50"
                  )}
                >
                  {isDone ? (
                    <span className="text-emerald-500 font-bold text-xs">✓</span>
                  ) : isCurrent ? (
                    <span className="inline-block size-2 rounded-full bg-primary animate-pulse" />
                  ) : (
                    <span className="inline-block size-2 rounded-full border border-muted-foreground/40" />
                  )}
                  <span className="truncate">{st.label}</span>
                </div>
              );
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function TaskView"""

if card_pattern.search(tv_text):
    tv_text = card_pattern.sub(new_card_code, tv_text)
    print("Replaced AgentProgressCard with collapsible thinking design")
else:
    print("AgentProgressCard pattern did not match")

# 4d. Pass toolEvents to AgentProgressCard in TaskView
tv_text = tv_text.replace(
    "model={stream.model}\n            elapsedSeconds={stream.elapsedSeconds}\n            onCancel={stream.cancel}\n          />",
    "model={stream.model}\n            elapsedSeconds={stream.elapsedSeconds}\n            toolEvents={stream.toolEvents}\n            onCancel={stream.cancel}\n          />"
)

taskview_path.write_text(tv_text, encoding="utf-8")
print("TaskView.tsx saved successfully")
