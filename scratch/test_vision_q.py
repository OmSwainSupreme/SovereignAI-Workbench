import asyncio
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"

async def test_vision():
    print("Testing Vision task with Question.png...")
    start_time = time.time()
    async with httpx.AsyncClient(timeout=300.0) as client:
        payload = {
            "task": "Read this image and solve ALL the questions completely.",
            "attachments": ["Question.png"],
        }
        resp = await client.post(f"{BASE_URL}/api/v1/tasks", json=payload)
        assert resp.status_code in (200, 201), f"Submit failed: {resp.text}"
        data = resp.json()
        task_id = data["id"]
        print(f"Task ID: {task_id}, Status: {data['status']}")

        last_phase = ""
        while True:
            await asyncio.sleep(2.0)
            try:
                poll = await client.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
                if poll.status_code != 200:
                    continue
                t = poll.json()
                st = t.get("status")
                ph = t.get("phase") or ""
                mod = t.get("model") or ""
                if ph != last_phase:
                    print(f"[{t.get('elapsedSeconds')}s] Phase: {ph} | Model: {mod}")
                    last_phase = ph
                if st in ("completed", "failed", "cancelled"):
                    print(f"\nFinal status: {st} in {time.time() - start_time:.1f}s")
                    print(f"Model: {mod}")
                    msgs = t.get("messages", [])
                    agent_msgs = [m for m in msgs if m.get("role") == "agent"]
                    if agent_msgs:
                        full_content = agent_msgs[-1].get("content", "")
                        print(f"Content length: {len(full_content)}")
                        print("\n=== FULL RESPONSE ===")
                        print(full_content)
                        print("=====================")
                    return t
            except Exception as e:
                print(f"Poll notice: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision())
