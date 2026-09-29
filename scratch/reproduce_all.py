import asyncio
import httpx
import time
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000/api/v1"

async def test_task(label: str, prompt: str):
    print(f"\n{'='*60}\nSTARTING {label}: {prompt[:80]}...\n{'='*60}", flush=True)
    async with httpx.AsyncClient(timeout=600.0) as client:
        # Submit task
        t_submit = time.time()
        res = await client.post(f"{BASE_URL}/tasks", json={"task": prompt})
        if res.status_code not in (200, 201):
            print(f"Submit FAILED: {res.status_code} {res.text}", flush=True)
            return
        task_data = res.json()
        task_id = task_data["id"]
        print(f"Submitted task_id={task_id}", flush=True)

        thinking_start = None
        thinking_end = None
        content_start = None
        content_end = None
        last_status = None
        last_phase = None
        last_phase_label = None
        last_thinking_len = 0
        last_content_len = 0
        thinking_updates = 0
        content_updates = 0

        while True:
            await asyncio.sleep(0.5)
            r = await client.get(f"{BASE_URL}/tasks/{task_id}")
            if r.status_code != 200:
                print(f"Poll error: {r.status_code}", flush=True)
                continue
            t = r.json()
            status = t.get("status")
            phase = t.get("phase")
            phase_label = t.get("phaseLabel")
            model = t.get("model")
            elapsed = time.time() - t_submit

            agent_msg = None
            for m in t.get("messages", []):
                if m.get("role") in ("agent", "assistant"):
                    agent_msg = m
                    break

            th = (agent_msg.get("thinking") or "") if agent_msg else ""
            ct = (agent_msg.get("content") or "") if agent_msg else ""

            if len(th) > last_thinking_len:
                if thinking_start is None:
                    thinking_start = time.time()
                    print(f"[{elapsed:5.1f}s] Thinking started!", flush=True)
                last_thinking_len = len(th)
                thinking_updates += 1

            if len(ct) > last_content_len:
                if thinking_start is not None and thinking_end is None:
                    thinking_end = time.time()
                    print(f"[{elapsed:5.1f}s] Thinking ended (duration: {thinking_end - thinking_start:.1f}s, chars: {len(th)})", flush=True)
                if content_start is None:
                    content_start = time.time()
                    print(f"[{elapsed:5.1f}s] Content generation started!", flush=True)
                last_content_len = len(ct)
                content_updates += 1

            if status != last_status or phase != last_phase or phase_label != last_phase_label:
                last_status = status
                last_phase = phase
                last_phase_label = phase_label
                print(f"[{elapsed:5.1f}s] Status: {status} | Phase: {phase} | Label: {phase_label} | Model: {model} | ThChars: {len(th)} | CtChars: {len(ct)}", flush=True)

            if status in ("completed", "failed", "cancelled"):
                total_duration = time.time() - t_submit
                print(f"\n--- RESULT FOR {label} ---")
                print(f"Final Status: {status}")
                print(f"Total Duration: {total_duration:.1f}s")
                print(f"Selected Model: {model}")
                print(f"Thinking Start: {thinking_start - t_submit if thinking_start else None}")
                print(f"Thinking Duration: {(thinking_end or time.time()) - thinking_start if thinking_start else 0:.1f}s")
                print(f"Content Start: {content_start - t_submit if content_start else None}")
                print(f"Thinking updates: {thinking_updates}, Content updates: {content_updates}")
                print(f"Final Thinking Length: {len(th)}")
                print(f"Final Content Length: {len(ct)}")
                print(f"Phase Label: {phase_label}")
                if status == "failed":
                    print(f"FAILED REASON: {t.get('error') or phase_label}")
                else:
                    print(f"Sample Content: {ct[:150]}...")
                return status == "completed"

async def main():
    prompts = [
        ("TEST A", "Explain abstraction in C++."),
        ("TEST B", "Explain abstraction, encapsulation, inheritance and polymorphism in C++ with examples."),
        ("TEST C", "Explain abstraction, encapsulation, inheritance and polymorphism in C++ in detail. Include definitions, real-world analogies, C++ examples, advantages, disadvantages, and a comparison table."),
    ]
    for label, p in prompts:
        await test_task(label, p)

if __name__ == "__main__":
    asyncio.run(main())
