import asyncio
import json
import time
import httpx

async def main():
    url = "http://127.0.0.1:11434/api/chat"
    prompt = "Explain abstraction in C++ in simple words."
    payload = {
        "model": "qwen3:4b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "options": {
            "num_ctx": 4096,
        }
    }

    print(f"Connecting to {url} with model {payload['model']}...", flush=True)
    start_time = time.time()
    chunk_idx = 0
    th_count = 0
    ct_count = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            print(f"HTTP Status: {response.status_code}", flush=True)
            async for line in response.aiter_lines():
                now = time.time()
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"JSON decode error: {e}", flush=True)
                    continue

                msg = data.get("message", {})
                th = msg.get("thinking", "")
                ct = msg.get("content", "")
                done = data.get("done", False)

                chunk_idx += 1
                if th:
                    th_count += 1
                    if th_count <= 5 or th_count % 20 == 0:
                        print(f"[{now - start_time:5.2f}s] Chunk {chunk_idx:3d} [THINKING #{th_count:3d}]: {repr(th)}", flush=True)
                if ct:
                    ct_count += 1
                    if ct_count <= 5 or ct_count % 20 == 0:
                        print(f"[{now - start_time:5.2f}s] Chunk {chunk_idx:3d} [CONTENT  #{ct_count:3d}]: {repr(ct)}", flush=True)

                if done:
                    print(f"[{now - start_time:5.2f}s] DONE signal! Total chunks: {chunk_idx}, Thinking: {th_count}, Content: {ct_count}", flush=True)
                    break

    print(f"Finished in {time.time() - start_time:.2f}s. Total thinking tokens: {th_count}, content tokens: {ct_count}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
