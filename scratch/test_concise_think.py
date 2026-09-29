import asyncio
import httpx
import time
import json

async def test():
    url = 'http://127.0.0.1:11434/api/chat'
    payload = {
        'model': 'qwen3:4b',
        'messages': [
            {'role': 'system', 'content': 'You are SovereignAI, a helpful technical AI assistant. Think concisely in 2-3 brief sentences, then provide a clear, simple answer.'},
            {'role': 'user', 'content': 'Explain abstraction in C++ in simple words.'}
        ],
        'stream': True,
        'options': {'num_ctx': 4096}
    }
    start = time.time()
    th_count = 0
    ct_count = 0
    print("Starting test...", flush=True)
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream('POST', url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.strip(): continue
                data = json.loads(line)
                msg = data.get('message', {})
                th = msg.get('thinking', '')
                ct = msg.get('content', '')
                if th:
                    th_count += 1
                    if th_count == 1:
                        print(f"First thinking token at {time.time() - start:.2f}s: {repr(th)}", flush=True)
                    elif th_count % 50 == 0:
                        print(f"Thinking token #{th_count} at {time.time() - start:.2f}s", flush=True)
                if ct:
                    ct_count += 1
                    if ct_count == 1:
                        print(f"First content token at {time.time() - start:.2f}s after {th_count} thinking tokens: {repr(ct)}", flush=True)
                    elif ct_count % 20 == 0:
                        print(f"Content token #{ct_count} at {time.time() - start:.2f}s", flush=True)
                if data.get('done'):
                    print(f"Done at {time.time() - start:.2f}s! Total thinking: {th_count}, content: {ct_count}", flush=True)

if __name__ == "__main__":
    asyncio.run(test())
