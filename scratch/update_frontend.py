import re
from pathlib import Path

# 1. Update streamReducer.ts
stream_reducer_path = Path(r"C:\lovable\src\features\workspace\streamReducer.ts")
content = stream_reducer_path.read_text(encoding="utf-8")

old_done = """        case "done":
          return {
            ...state,
            phase: "completed",
            phaseLabel: "Completed",
            messages: state.messages.map((m) => ({ ...m, pending: false })),
            status:
              state.status === "running" || state.status === "queued"
                ? "completed"
                : state.status,
          };"""

new_done = """        case "done": {
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
        }"""

assert old_done in content, "old_done not found in streamReducer.ts"
content = content.replace(old_done, new_done)
stream_reducer_path.write_text(content, encoding="utf-8")
print("streamReducer.ts updated successfully")

# 2. Update pollingAdapter.ts
polling_adapter_path = Path(r"C:\lovable\src\services\stream\pollingAdapter.ts")
content = polling_adapter_path.read_text(encoding="utf-8")

old_failed = """      if (task.status === "failed") {
        yield { type: "error", message: "The task failed to complete." };
        return;
      }"""

new_failed = """      if (task.status === "failed") {
        yield { type: "error", message: task.phaseLabel || "The task failed to complete." };
        return;
      }"""

assert old_failed in content, "old_failed not found in pollingAdapter.ts"
content = content.replace(old_failed, new_failed)
polling_adapter_path.write_text(content, encoding="utf-8")
print("pollingAdapter.ts updated successfully")

# 3. Update TaskView.tsx
task_view_path = Path(r"C:\lovable\src\features\workspace\TaskView.tsx")
content = task_view_path.read_text(encoding="utf-8")

# Add FileText, ImageIcon to lucide-react imports
old_import = 'import { ArrowUp, Paperclip, Square, Wrench, X, Loader2 } from "lucide-react";'
new_import = 'import { ArrowUp, Paperclip, Square, Wrench, X, Loader2, FileText, ImageIcon } from "lucide-react";'
assert old_import in content, "old_import not found in TaskView.tsx"
content = content.replace(old_import, new_import)

# Update message attachment rendering
old_msg_attach = """            {message.attachments?.length ? (
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
            ) : null}"""

new_msg_attach = """            {message.attachments?.length ? (
              <ul className="mt-2.5 flex flex-wrap gap-1.5">
                {message.attachments.map((name) => {
                  const isImage = /\\.(png|jpe?g|webp|bmp|gif|tiff)$/i.test(name);
                  return (
                    <li
                      key={name}
                      className="text-mono flex items-center gap-1.5 rounded-md border border-border/70 bg-background/80 px-2.5 py-1 text-xs text-muted-foreground"
                    >
                      {isImage ? (
                        <ImageIcon className="size-3.5 text-primary" aria-hidden />
                      ) : (
                        <FileText className="size-3.5 text-muted-foreground" aria-hidden />
                      )}
                      <span>{name}</span>
                    </li>
                  );
                })}
              </ul>
            ) : null}"""

assert old_msg_attach in content, "old_msg_attach not found in TaskView.tsx"
content = content.replace(old_msg_attach, new_msg_attach)

# Update useEffect for taskId and stream.status in TaskView
old_focus_effect = """  useEffect(() => {
    inputRef.current?.focus();
  }, [taskId]);"""

new_focus_effect = """  useEffect(() => {
    inputRef.current?.focus();
    // Reset composer attachments when switching tasks
    setAttachments([]);
    setAttachError(null);
    if (fileRef.current) {
      fileRef.current.value = "";
    }
  }, [taskId]);

  useEffect(() => {
    if (stream.status === "completed" || stream.status === "failed" || stream.status === "cancelled") {
      setAttachments([]);
      setAttachError(null);
      if (fileRef.current) {
        fileRef.current.value = "";
      }
    }
  }, [stream.status]);"""

assert old_focus_effect in content, "old_focus_effect not found in TaskView.tsx"
content = content.replace(old_focus_effect, new_focus_effect)

# Update composer attachment list
old_composer_attach = """          {attachments.length > 0 ? (
            <ul className="mb-2 flex flex-wrap gap-1.5">
              {attachments.map((file) => (
                <li
                  key={file.name}
                  className="text-mono flex items-center gap-1.5 rounded border border-border bg-background px-2 py-1 text-xs text-muted-foreground"
                >
                  {file.name}
                  <button
                    type="button"
                    aria-label={`Remove ${file.name}`}
                    onClick={() => clearAttachment(file)}
                    className="rounded-sm hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <X className="size-3" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          ) : null}"""

new_composer_attach = """          {attachments.length > 0 ? (
            <ul className="mb-2 flex flex-wrap gap-2">
              {attachments.map((file) => {
                const isImage =
                  file.type.startsWith("image/") ||
                  /\\.(png|jpe?g|webp|bmp|gif|tiff)$/i.test(file.name);
                const previewUrl = isImage ? URL.createObjectURL(file) : null;
                return (
                  <li
                    key={file.name}
                    className="flex items-center gap-2 rounded-md border border-border/80 bg-surface/95 px-2.5 py-1.5 text-xs text-foreground shadow-sm transition-all"
                  >
                    {isImage && previewUrl ? (
                      <img
                        src={previewUrl}
                        alt={file.name}
                        className="size-8 rounded object-cover border border-border/60"
                      />
                    ) : (
                      <div className="flex size-8 shrink-0 items-center justify-center rounded bg-muted/60 text-muted-foreground">
                        <FileText className="size-4 text-primary/80" aria-hidden />
                      </div>
                    )}
                    <div className="flex flex-col max-w-[180px] min-w-0">
                      <span className="truncate font-medium text-xs text-foreground" title={file.name}>
                        {file.name}
                      </span>
                      <span className="text-[10px] text-muted-foreground">
                        {isImage ? "Image" : "Document"} • {(file.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                    <button
                      type="button"
                      aria-label={`Remove ${file.name}`}
                      onClick={() => clearAttachment(file)}
                      className="ml-1 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <X className="size-3.5" aria-hidden />
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : null}"""

assert old_composer_attach in content, "old_composer_attach not found in TaskView.tsx"
content = content.replace(old_composer_attach, new_composer_attach)

task_view_path.write_text(content, encoding="utf-8")
print("TaskView.tsx updated successfully")
