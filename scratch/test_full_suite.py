import asyncio
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"

async def test_case(name: str, task_text: str, attachments: list[str] = None):
    print(f"\n--- Testing: {name} ---")
    start_time = time.time()
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Submit
        payload = {"task": task_text}
        if attachments:
            payload["attachments"] = attachments
        resp = await client.post(f"{BASE_URL}/api/v1/tasks", json=payload)
        assert resp.status_code in (200, 201), f"Submit failed: {resp.text}"
        data = resp.json()
        task_id = data["id"]
        print(f"Task ID: {task_id}, Initial Status: {data['status']}")

        # Poll
        last_phase = ""
        while True:
            await asyncio.sleep(1.0)
            poll_resp = await client.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
            assert poll_resp.status_code == 200, f"Poll failed: {poll_resp.text}"
            task = poll_resp.json()
            status = task["status"]
            phase = task.get("phase") or ""
            model = task.get("model") or ""
            if phase != last_phase:
                print(f"Phase: {phase} | Model: {model} | Elapsed: {task.get('elapsedSeconds')}s")
                last_phase = phase
            if status in ("completed", "failed", "cancelled"):
                total_duration = time.time() - start_time
                print(f"Final Status: {status} in {total_duration:.1f}s")
                print(f"Selected Model: {task.get('model')}")
                messages = task.get("messages", [])
                agent_msgs = [m for m in messages if m.get("role") == "agent"]
                if agent_msgs:
                    msg = agent_msgs[-1]
                    thinking = msg.get("thinking")
                    content = msg.get("content")
                    print(f"Has thinking stream: {bool(thinking)} (length: {len(thinking) if thinking else 0})")
                    print(f"Content length: {len(content) if content else 0}")
                    print(f"Content Preview:\n{content[:300]}...")
                return {
                    "name": name,
                    "status": status,
                    "model": task.get("model"),
                    "duration": total_duration,
                    "thinking": bool(agent_msgs and agent_msgs[-1].get("thinking")),
                    "content": agent_msgs[-1].get("content") if agent_msgs else "",
                    "tools": [t.get("label") for t in task.get("toolEvents", [])],
                }

async def main():
    results = []
    # 1. Simple question
    results.append(await test_case("Simple General Question", "Explain abstraction in C++ in simple words."))
    
    # 2. Coding question
    results.append(await test_case("Coding Question", "Write a C++ program to reverse a string."))

    # 3. Document TXT summary
    results.append(await test_case("Document TXT Summary", "Summarize this file in 3 bullet points.", ["sample.txt"]))

    # 4. Vision / OCR
    results.append(await test_case("Vision / OCR Question.png", "Read this image and solve ALL the questions completely.", ["Question.png"]))

    print("\n================== FULL SUITE SUMMARY ==================")
    for r in results:
        print(f"[{r['status'].upper()}] {r['name']} | Model: {r['model']} | Latency: {r['duration']:.1f}s | Thinking: {r['thinking']} | Tools: {r['tools']}")

if __name__ == "__main__":
    asyncio.run(main())
