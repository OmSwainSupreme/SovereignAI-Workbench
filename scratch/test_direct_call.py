import httpx
import time

def main():
    start = time.time()
    print("Calling Ollama...", flush=True)
    resp = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "qwen3:4b",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": False
        },
        timeout=120.0
    )
    print(f"Done in {time.time() - start:.2f}s: {resp.json().get('message', {})}", flush=True)

if __name__ == "__main__":
    main()
