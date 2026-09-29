import asyncio
import httpx
import time
import json
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

async def test():
    url = "http://127.0.0.1:11434/api/chat"
    prompt = "Explain abstraction in C++ in simple words."
    payload = {
        "model": "qwen3:4b",
        "messages": [
            {
                "role": "system",
                "content": "You are SovereignAI, an expert technical AI assistant. Think concisely in 2-3 brief reasoning steps before providing the final clear answer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True,
        "options": {
            "num_ctx": 4096,
            "num_predict": 700,
            "temperature": 0.3,
        }
    }
    start = time.time()
    th_count = 0
    ct_count = 0
    first_th = None
    first_ct = None

    print(f"Connecting to Ollama for prompt: '{prompt}'...", flush=True)
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            print(f"HTTP Status: {resp.status_code}", flush=True)
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                msg = data.get("message", {})
                th = msg.get("thinking", "")
                ct = msg.get("content", "")

                if th:
                    th_count += 1
                    if first_th is None:
                        first_th = time.time()
                        print(f"[{first_th - start:5.2f}s] FIRST THINKING CHUNK: {repr(th)}", flush=True)
                    elif th_count % 30 == 0:
                        print(f"[{time.time() - start:5.2f}s] Thinking chunk #{th_count}", flush=True)

                if ct:
                    ct_count += 1
                    if first_ct is None:
                        first_ct = time.time()
                        print(f"[{first_ct - start:5.2f}s] FIRST CONTENT CHUNK (after {th_count} thinking chunks): {repr(ct)}", flush=True)
                    elif ct_count % 30 == 0:
                        print(f"[{time.time() - start:5.2f}s] Content chunk #{ct_count}", flush=True)

                if data.get("done"):
                    print(f"[{time.time() - start:5.2f}s] STREAM DONE! Total thinking: {th_count}, content: {ct_count}", flush=True)
                    break

if __name__ == "__main__":
    asyncio.run(test())
