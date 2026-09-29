import sys
sys.stdout.reconfigure(encoding='utf-8')

def update_task_view():
    path = r"C:\lovable\src\features\workspace\TaskView.tsx"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_snippet = """function MessageBubble({ message }: { message: ChatMessage }) {
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
            </button>"""

    new_snippet = """function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [userToggled, setUserToggled] = useState<boolean | null>(null);
  const hasThinking = Boolean(message.thinking && message.thinking.trim().length > 0);
  const isThinkingActive = Boolean(message.pending && !message.content);
  const thinkingOpen = userToggled !== null ? userToggled : isThinkingActive;

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div className={cn("max-w-[46rem] space-y-2", isUser && "flex flex-col items-end")}>
        {!isUser && hasThinking ? (
          <div className="w-full">
            <button
              type="button"
              onClick={() => setUserToggled((prev) => (prev !== null ? !prev : !isThinkingActive))}
              className="inline-flex items-center gap-1.5 rounded-md border border-border/70 bg-surface/90 px-2.5 py-1 text-xs font-medium text-muted-foreground hover:bg-surface hover:text-foreground transition-colors cursor-pointer"
              aria-expanded={thinkingOpen}
            >
              <span className="size-1.5 rounded-full bg-primary/80 animate-pulse" />
              <span>Thinking</span>
              <span className="text-[10px] font-mono">{thinkingOpen ? "▴" : "▾"}</span>
            </button>"""

    if old_snippet in content:
        new_content = content.replace(old_snippet, new_snippet, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated TaskView.tsx successfully!")
    else:
        print("old_snippet not found in TaskView.tsx")

if __name__ == "__main__":
    update_task_view()
