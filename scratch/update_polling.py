from pathlib import Path

path = Path(r"C:\lovable\src\services\stream\pollingAdapter.ts")
content = path.read_text(encoding="utf-8")

old_const = "const POLL_INTERVAL_MS = 500;"
new_const = """const INITIAL_POLL_INTERVAL_MS = 100;
const ACTIVE_POLL_INTERVAL_MS = 150;
const POLL_INTERVAL_MS = 300;"""

assert old_const in content, "old_const not found"
content = content.replace(old_const, new_const, 1)

old_init = 'let lastPhaseLabel = "";'
new_init = 'let lastPhaseLabel = "";\n    let pollIteration = 0;'

assert old_init in content, "old_init not found"
content = content.replace(old_init, new_init, 1)

old_sleep = "await abortableSleep(POLL_INTERVAL_MS, signal);"
new_sleep = """pollIteration++;
      const currentInterval =
        pollIteration <= 2
          ? INITIAL_POLL_INTERVAL_MS
          : task.status === "running"
            ? ACTIVE_POLL_INTERVAL_MS
            : POLL_INTERVAL_MS;
      await abortableSleep(currentInterval, signal);"""

assert old_sleep in content, "old_sleep not found"
content = content.replace(old_sleep, new_sleep, 1)

path.write_text(content, encoding="utf-8")
print("Successfully updated C:\\lovable\\src\\services\\stream\\pollingAdapter.ts")
