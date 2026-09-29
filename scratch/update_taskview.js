const fs = require('fs');

const file = 'C:/lovable/src/features/workspace/TaskView.tsx';
let content = fs.readFileSync(file, 'utf8');

// 1. Replace MessageBubble and AgentProgressCard
const mbStart = content.indexOf('function MessageBubble');
const tvStart = content.indexOf('export function TaskView');

if (mbStart !== -1 && tvStart !== -1) {
  const newMiddle = `function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const hasThinking = Boolean(message.thinking && message.thinking.trim().length > 0);

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div className={cn("max-w-[46rem] space-y-2", isUser && "flex flex-col items-end")}>
        {!isUser && hasThinking ? (
          <div className="w-full">
            <button
              type="button"
              onClick={() => setThinkingOpen((prev) => !prev)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border/70 bg-surface/90 px-2.5 py-1 text-xs font-medium text-muted-foreground hover:bg-surface hover:text-foreground transition-colors cursor-pointer"
              aria-expanded={thinkingOpen}
            >
              <span className="size-1.5 rounded-full bg-primary/80 animate-pulse" />
              <span>Thinking</span>
              <span className="text-[10px] font-mono">{thinkingOpen ? "▴" : "▾"}</span>
            </button>
            {thinkingOpen ? (
              <div className="mt-1.5 max-h-64 overflow-y-auto rounded-lg border border-border/70 bg-surface/60 p-3 text-xs leading-relaxed text-muted-foreground font-mono whitespace-pre-wrap">
                {message.thinking}
                {message.pending && !message.content ? (
                  <span className="ml-0.5 inline-block h-3.5 w-1 animate-pulse bg-primary align-text-bottom" />
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}

        {Boolean(message.content || !hasThinking) ? (
          <div
            className={cn(
              "rounded-xl px-4 py-3 text-sm leading-relaxed",
              isUser
                ? "bg-primary/15 text-foreground"
                : "border border-border/70 bg-surface text-foreground",
            )}
          >
            <p className="whitespace-pre-wrap">
              {message.content}
              {message.pending && message.content ? (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-primary align-text-bottom" />
              ) : null}
            </p>
            {message.attachments?.length ? (
              <ul className="mt-2.5 flex flex-wrap gap-1.5">
                {message.attachments.map((name) => (
                  <li
                    key={name}
                    className="text-mono rounded border border-border/70 bg-background/60 px-2 py-0.5 text-xs text-muted-foreground"
                  >
                    {name}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}

        {message.sources?.length ? (
          <SourceList sources={message.sources} grounded={message.grounded ?? false} />
        ) : null}
      </div>
    </div>
  );
}

function AgentProgressCard({
  status,
  phase,
  phaseLabel,
  model,
  elapsedSeconds,
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

  useEffect(() => {
    if (status !== "running" && status !== "queued") return;
    const start = Date.now() - (elapsedSeconds ? elapsedSeconds * 1000 : 0);
    const interval = setInterval(() => {
      setElapsed(Math.max(1, Math.round((Date.now() - start) / 1000)));
    }, 1000);
    return () => clearInterval(interval);
  }, [status, elapsedSeconds]);

  if (status !== "running" && status !== "queued") return null;

  return (
    <div className="rounded-xl border border-primary/30 bg-surface/90 px-4 py-3 shadow-md backdrop-blur">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <span className="relative flex size-2.5 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
            <span className="relative inline-flex size-2.5 rounded-full bg-primary" />
          </span>
          <span className="truncate text-sm font-medium text-foreground">
            {phaseLabel || (status === "queued" ? "Queued in task runner..." : "Agent working on request...")}
          </span>
        </div>
        <div className="flex items-center gap-3 shrink-0">
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
    </div>
  );
}

`;
  content = content.slice(0, mbStart) + newMiddle + content.slice(tvStart);
}

// 2. Clean placeholder in input box
content = content.replace(
  'placeholder={isBusy ? "AI is working on your current task..." : "Describe a task."}',
  'placeholder={isBusy ? "Task in progress..." : "Describe a task..."}'
);

// 3. Clean helper text
content = content.replace(
  '{isBusy\n              ? "A task is running in the sandbox. You can cancel with the stop button."\n              : "Files are validated by the backend · Enter to send · Shift+Enter for a new line"}',
  '"Files are validated by the backend · Enter to send · Shift+Enter for a new line"'
);

fs.writeFileSync(file, content, 'utf8');
console.log('TaskView.tsx updated successfully');
