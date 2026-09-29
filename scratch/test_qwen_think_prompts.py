import httpx, time, json

client = httpx.Client(base_url="http://127.0.0.1:11434", timeout=60)
prompt = "Summarize sample.txt in 2 short bullet points."

# Test A: Instruct to output final answer directly
t0 = time.perf_counter()
rA = client.post("/api/chat", json={
    "model": "qwen3:4b",
    "messages": [
        {"role": "system", "content": "You are SovereignAI. Provide the final summary immediately and directly. Do not use <think> tags or internal reasoning."},
        {"role": "user", "content": prompt}
    ],
    "options": {"num_ctx": 2048, "num_predict": 150, "temperature": 0.3},
    "keep_alive": "60m",
    "stream": False
}).json()
dtA = time.perf_counter() - t0
msgA = rA.get("message", {})
th_lenA = len(msgA.get("thinking", ""))
ct_lenA = len(msgA.get("content", ""))
print(f"Test A: time={dtA:.2f}s, think_len={th_lenA}, content_len={ct_lenA}")
print("Content A:", msgA.get("content"))
print("Thinking A snippet:", msgA.get("thinking", "")[:100])
