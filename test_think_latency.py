import httpx
import time

client = httpx.Client(base_url="http://127.0.0.1:11434", timeout=300)
prompt = "Summarize in 3 bullet points:\nNetwork security protects networks and data from unauthorized access. Cloud security protects cloud infrastructure and SaaS applications. Application security identifies vulnerabilities in source code."

print("Testing qwen3:4b with default prompt...")
t0 = time.time()
r1 = client.post("/api/chat", json={
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False
})
dt1 = time.time() - t0
data1 = r1.json()
msg1 = data1.get("message", {}).get("content", "")
has_think = "<think>" in msg1
print(f"qwen3:4b latency: {dt1:.1f}s, has_think: {has_think}, total_len: {len(msg1)}")
if has_think:
    think_part = msg1.split("</think>")[0]
    print(f"think length: {len(think_part)} chars")

print("\nTesting qwen3:4b with explicit no-think instruction...")
t0 = time.time()
r2 = client.post("/api/chat", json={
    "model": "qwen3:4b",
    "messages": [
        {"role": "system", "content": "/set no_thinking\nYou are a helpful concise assistant. Directly output the answer without thinking tags."},
        {"role": "user", "content": prompt}
    ],
    "stream": False
})
dt2 = time.time() - t0
data2 = r2.json()
msg2 = data2.get("message", {}).get("content", "")
print(f"qwen3:4b no-think latency: {dt2:.1f}s, has_think: {'<think>' in msg2}, total_len: {len(msg2)}")
