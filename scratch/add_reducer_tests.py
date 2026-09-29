from pathlib import Path

test_file = Path(r"C:\lovable\src\features\workspace\streamReducer.test.ts")
content = test_file.read_text(encoding="utf-8")

new_tests = """  it("fails task if done event arrives with only thinking and no assistant content", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "thinking-delta", messageId: "m1", text: "Reasoning only..." },
      { type: "done" },
    ]);

    expect(state.status).toBe("failed");
    expect(state.phase).toBe("failed");
    expect(state.phaseLabel).toBe("Generation interrupted: No final answer produced");
    expect(state.messages[0]?.pending).toBe(false);
  });

  it("fails task if done event arrives with empty content", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "token-delta", messageId: "m1", text: "   " },
      { type: "done" },
    ]);

    expect(state.status).toBe("failed");
    expect(state.phase).toBe("failed");
    expect(state.phaseLabel).toBe("Generation interrupted: No final answer produced");
  });

  it("completes task if done event arrives with valid assistant content", () => {
    const state = run([
      { type: "status", status: "running" },
      { type: "thinking-delta", messageId: "m1", text: "Reasoning..." },
      { type: "token-delta", messageId: "m1", text: "Valid final answer." },
      { type: "done" },
    ]);

    expect(state.status).toBe("completed");
    expect(state.phase).toBe("completed");
    expect(state.phaseLabel).toBe("Completed");
    expect(state.messages[0]?.content).toBe("Valid final answer.");
    expect(state.messages[0]?.pending).toBe(false);
  });
});
"""

content = content.rstrip()
if content.endswith("});"):
    content = content[:-3] + new_tests
    test_file.write_text(content, encoding="utf-8")
    print("Added regression tests to streamReducer.test.ts")
else:
    print("Could not find ending of describe block")
