from pathlib import Path

p = Path(r"C:\lovable\src\features\workspace\TaskView.tsx")
text = p.read_text(encoding="utf-8")

start_str = "function AgentProgressCard({"
end_str = "export function TaskView({"

start_idx = text.find(start_str)
end_idx = text.find(end_str)

if start_idx != -1 and end_idx != -1:
    new_card = """function AgentProgressCard({
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
          <span className="text-[10px]">{isExpanded ? "\u25b4" : "\u25be"}</span>
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
                    <span className="text-emerald-500 font-bold text-xs">\u2713</span>
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

"""
    updated = text[:start_idx] + new_card + text[end_idx:]
    p.write_text(updated, encoding="utf-8")
    print("Successfully patched AgentProgressCard in TaskView.tsx")
else:
    print(f"Error: start_idx={start_idx}, end_idx={end_idx}")
