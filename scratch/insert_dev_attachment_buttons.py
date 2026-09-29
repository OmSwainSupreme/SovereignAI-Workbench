from pathlib import Path

task_view_path = Path(r"C:\lovable\src\features\workspace\TaskView.tsx")
content = task_view_path.read_text(encoding="utf-8")

old_paperclip = """            <Button
              type="button"
              size="icon"
              variant="ghost"
              aria-label="Attach files"
              disabled={isBusy}
              onClick={() => fileRef.current?.click()}
            >
              <Paperclip className="size-4" aria-hidden />
            </Button>"""

new_buttons = """            <Button
              type="button"
              size="icon"
              variant="ghost"
              aria-label="Attach files"
              disabled={isBusy}
              onClick={() => fileRef.current?.click()}
            >
              <Paperclip className="size-4" aria-hidden />
            </Button>
            {import.meta.env.DEV ? (
              <div className="flex items-center gap-1 self-center">
                <button
                  type="button"
                  data-testid="attach-txt-btn"
                  title="Attach sample.txt for testing"
                  disabled={isBusy}
                  className="rounded border border-border/80 bg-surface/80 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-surface hover:text-foreground"
                  onClick={() => {
                    const file = new File(
                      [
                        "SovereignAI Workbench is an autonomous local AI agent architecture.\\n" +
                        "It runs entirely on-premise without sending private data to external cloud providers.\\n" +
                        "Key features include local model routing, deterministic policy enforcement, Docker sandboxing, and audit logging.\\n" +
                        "It supports multimodality including vision and OCR with qwen2.5vl:3b, code execution with qwen2.5-coder:3b, and general reasoning with qwen3:4b."
                      ],
                      "sample.txt",
                      { type: "text/plain" }
                    );
                    setAttachments((a) => [...a, file]);
                  }}
                >
                  +TXT
                </button>
                <button
                  type="button"
                  data-testid="attach-png-btn"
                  title="Attach Question.png for testing"
                  disabled={isBusy}
                  className="rounded border border-border/80 bg-surface/80 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-surface hover:text-foreground"
                  onClick={async () => {
                    try {
                      const res = await fetch("/api/v1/files/Question.png");
                      const blob = await res.blob();
                      const file = new File([blob], "Question.png", { type: "image/png" });
                      setAttachments((a) => [...a, file]);
                    } catch (e) {
                      console.error(e);
                    }
                  }}
                >
                  +PNG
                </button>
                <button
                  type="button"
                  data-testid="attach-jpg-btn"
                  title="Attach test.jpg for testing"
                  disabled={isBusy}
                  className="rounded border border-border/80 bg-surface/80 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-surface hover:text-foreground"
                  onClick={async () => {
                    try {
                      const res = await fetch("/api/v1/files/test.jpg");
                      const blob = await res.blob();
                      const file = new File([blob], "test.jpg", { type: "image/jpeg" });
                      setAttachments((a) => [...a, file]);
                    } catch (e) {
                      console.error(e);
                    }
                  }}
                >
                  +JPG
                </button>
                <button
                  type="button"
                  data-testid="attach-jpeg-btn"
                  title="Attach test.jpeg for testing"
                  disabled={isBusy}
                  className="rounded border border-border/80 bg-surface/80 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-surface hover:text-foreground"
                  onClick={async () => {
                    try {
                      const res = await fetch("/api/v1/files/test.jpeg");
                      const blob = await res.blob();
                      const file = new File([blob], "test.jpeg", { type: "image/jpeg" });
                      setAttachments((a) => [...a, file]);
                    } catch (e) {
                      console.error(e);
                    }
                  }}
                >
                  +JPEG
                </button>
              </div>
            ) : null}"""

assert old_paperclip in content, "old_paperclip not found in TaskView.tsx"
content = content.replace(old_paperclip, new_buttons)
task_view_path.write_text(content, encoding="utf-8")
print("Inserted dev test attachment buttons into TaskView.tsx")
