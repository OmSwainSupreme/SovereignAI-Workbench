import httpx
import time

client = httpx.Client(base_url="http://127.0.0.1:11434", timeout=300)
prompt = """Topics – Basics of C++, Control Structures
1. Hello World + control characters
2. Celsius → Fahrenheit
3. Gross salary calculation
4. Steel grading using nested if-else / ladder / else-if
5. Library fine calculation"""

for max_tok in [400, 1500, 2500]:
    t0 = time.time()
    res = client.post("/api/chat", json={
        "model": "qwen2.5-coder:3b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "num_predict": max_tok,
            "temperature": 0.3
        },
        "stream": False
    })
    dt = time.time() - t0
    data = res.json()
    msg = data.get("message", {}).get("content", "")
    done_reason = data.get("done_reason")
    eval_count = data.get("eval_count", 0)
    print(f"\n--- max_tokens={max_tok} ---")
    print(f"Latency: {dt:.1f}s, Tokens: {eval_count}, Done Reason: {done_reason}")
    print(f"Has Q1: {'1.' in msg}, Has Q2: {'2.' in msg}, Has Q3: {'3.' in msg}, Has Q4: {'4.' in msg}, Has Q5: {'5.' in msg}")
    print("Tail of response (last 200 chars):")
    print(msg[-200:])
