from pathlib import Path

task_view_path = Path(r"C:\lovable\src\features\workspace\TaskView.tsx")
content = task_view_path.read_text(encoding="utf-8")

old_target = "  const isBusy = stream.isStreaming || stream.status === \"queued\" || stream.status === \"running\" || isSubmitting;"

hook_code = """  const isBusy = stream.isStreaming || stream.status === \"queued\" || stream.status === \"running\" || isSubmitting;

  useEffect(() => {
    if (typeof window !== "undefined") {
      (window as unknown as { __testAttachFile?: (name: string, content: string, type?: string) => void }).__testAttachFile = (
        name: string,
        content: string,
        type: string = "text/plain",
      ) => {
        const file = new File([content], name, { type });
        setAttachments((a) => [...a, file]);
      };
    }
  }, []);"""

assert old_target in content, "old_target not found in TaskView.tsx"
content = content.replace(old_target, hook_code)
task_view_path.write_text(content, encoding="utf-8")
print("Added window.__testAttachFile development hook to TaskView.tsx")
