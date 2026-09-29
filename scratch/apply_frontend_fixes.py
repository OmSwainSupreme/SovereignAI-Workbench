import os

def update_polling_adapter():
    path = r"C:\lovable\src\services\stream\pollingAdapter.ts"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = """        if (message.pending) {
          if (message.content.length > prevText.length) {
            const delta = message.content.slice(prevText.length);
            lastContentByMsg.set(message.id, message.content);
            yield { type: "token-delta", messageId: message.id, text: delta };
          }
        }"""

    replacement = """        if (message.pending) {
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
        }"""

    if target in content:
        new_content = content.replace(target, replacement, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated pollingAdapter.ts successfully.")
    else:
        print("Target not found in pollingAdapter.ts (may already be updated).")

def update_stream_reducer():
    path = r"C:\lovable\src\features\workspace\streamReducer.ts"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = """        case "token-delta":
          return {
            ...state,
            messages: applyDelta(state.messages, event.messageId, event.text),
          };"""

    replacement = """        case "token-delta":
          return {
            ...state,
            messages: applyDelta(state.messages, event.messageId, event.text),
          };
        case "thinking-delta":
          return {
            ...state,
            messages: applyThinkingDelta(state.messages, event.messageId, event.text),
          };"""

    if target in content:
        new_content = content.replace(target, replacement, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated streamReducer.ts successfully.")
    else:
        print("Target not found in streamReducer.ts (may already be updated).")

def update_task_view():
    path = r"C:\lovable\src\features\workspace\TaskView.tsx"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_bubble = """function MessageBubble({ message }: { message: ChatMessage }) {
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
              <span className="text-[10px] font-mono">{thinkingOpen ? "?" : "?"}</span>
            </button>"""

    new_bubble = """function MessageBubble({ message }: { message: ChatMessage }) {
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
              <span className="text-[10px] font-mono">{thinkingOpen ? "▲" : "▼"}</span>
            </button>"""

    if old_bubble in content:
        new_content = content.replace(old_bubble, new_bubble, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated TaskView.tsx successfully.")
    else:
        print("Old bubble not found in TaskView.tsx, checking partial match...")
        # Check if ? exists in glyph
        idx = content.find("function MessageBubble")
        if idx != -1:
            end_idx = content.find("</button>", idx)
            print("Current MessageBubble header snippet:\n", content[idx:end_idx+9])

if __name__ == "__main__":
    update_polling_adapter()
    update_stream_reducer()
    update_task_view()
