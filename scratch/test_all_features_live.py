import asyncio
import io
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        print("==================================================")
        print("1. Testing Root, Health, Info, Models Status")
        print("==================================================")
        r = await client.get(f"{BASE_URL}/")
        assert r.status_code == 200, f"Root failed: {r.status_code} {r.text}"
        print(f"PASS: GET / -> {r.json()}")

        r = await client.get(f"{BASE_URL}/health")
        assert r.status_code == 200, f"Health failed: {r.status_code} {r.text}"
        print(f"PASS: GET /health -> {r.json()}")

        r = await client.get(f"{BASE_URL}/info")
        assert r.status_code == 200, f"Info failed: {r.status_code} {r.text}"
        print(f"PASS: GET /info -> {r.json()['name']}")

        r = await client.get(f"{BASE_URL}/api/v1/models/status")
        assert r.status_code == 200, f"Models status failed: {r.status_code} {r.text}"
        print(f"PASS: GET /api/v1/models/status -> provider={r.json().get('provider')} available={r.json().get('available')}")

        print("\n==================================================")
        print("2. Testing Files API")
        print("==================================================")
        r = await client.get(f"{BASE_URL}/api/v1/files")
        assert r.status_code == 200, f"List files failed: {r.status_code} {r.text}"
        files_data = r.json()
        print(f"PASS: GET /api/v1/files -> {len(files_data.get('files', []))} files listed")

        # Test upload
        test_filename = f"test_live_{int(time.time())}.txt"
        files = {"file": (test_filename, b"Hello SovereignAI live test content", "text/plain")}
        r = await client.post(f"{BASE_URL}/api/v1/files", files=files)
        assert r.status_code in (200, 201), f"Upload failed: {r.status_code} {r.text}"
        print(f"PASS: POST /api/v1/files -> uploaded {test_filename}")

        # Test download
        r = await client.get(f"{BASE_URL}/api/v1/files/{test_filename}/download")
        assert r.status_code == 200, f"Download failed: {r.status_code} {r.text}"
        assert b"Hello SovereignAI live test content" in r.content
        print(f"PASS: GET /api/v1/files/{test_filename}/download -> content verified")

        print("\n==================================================")
        print("3. Testing Code Execution API")
        print("==================================================")
        code_payload = {"code": "x = 40 + 2\nprint(f'Result: {x}')"}
        r = await client.post(f"{BASE_URL}/api/v1/code/execute", json=code_payload)
        assert r.status_code == 200, f"Code execute failed: {r.status_code} {r.text}"
        code_res = r.json()
        print(f"PASS: POST /api/v1/code/execute -> success={code_res.get('success')} stdout={code_res.get('stdout').strip()}")

        print("\n==================================================")
        print("4. Testing Knowledge API")
        print("==================================================")
        r = await client.get(f"{BASE_URL}/api/v1/knowledge/collections")
        assert r.status_code == 200, f"Knowledge collections failed: {r.status_code} {r.text}"
        print(f"PASS: GET /api/v1/knowledge/collections -> {r.json()}")

        r = await client.post(f"{BASE_URL}/api/v1/knowledge/search", json={"query": "test query", "top_k": 3})
        assert r.status_code == 200, f"Knowledge search failed: {r.status_code} {r.text}"
        print(f"PASS: POST /api/v1/knowledge/search -> count={r.json().get('count')}")

        print("\n==================================================")
        print("5. Testing Activity / Audit API")
        print("==================================================")
        r = await client.get(f"{BASE_URL}/api/v1/activity")
        assert r.status_code == 200, f"Activity failed: {r.status_code} {r.text}"
        print(f"PASS: GET /api/v1/activity -> {len(r.json().get('items', []))} audit items")

        print("\n==================================================")
        print("6. Testing Tasks API (Create, Poll, Complete)")
        print("==================================================")
        task_payload = {"task": "What is 15 * 14? Answer with just the number."}
        r = await client.post(f"{BASE_URL}/api/v1/tasks", json=task_payload)
        assert r.status_code in (200, 201), f"Create task failed: {r.status_code} {r.text}"
        task_id = r.json()["id"]
        print(f"PASS: POST /api/v1/tasks -> task_id={task_id}")

        # Poll task
        start_time = time.time()
        final_task = None
        while time.time() - start_time < 90:
            await asyncio.sleep(1.0)
            r = await client.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
            assert r.status_code == 200, f"Poll task failed: {r.status_code} {r.text}"
            task_data = r.json()
            status = task_data.get("status")
            phase = task_data.get("phase")
            if status in ("completed", "failed"):
                final_task = task_data
                break

        assert final_task is not None, "Task did not complete within timeout"
        print(f"PASS: GET /api/v1/tasks/{task_id} -> status={final_task['status']} model={final_task.get('model')}")
        agent_msgs = [m for m in final_task.get("messages", []) if m.get("role") == "agent"]
        if agent_msgs:
            print(f"Assistant Content: {agent_msgs[-1].get('content', '').strip()[:200]}")

        print("\n==================================================")
        print("7. Testing Tasks API Cancellation")
        print("==================================================")
        task_cancel_payload = {"task": "Write a long essay on the history of computers from 1800 to 2026."}
        r = await client.post(f"{BASE_URL}/api/v1/tasks", json=task_cancel_payload)
        cancel_id = r.json()["id"]
        # Immediately cancel
        r = await client.post(f"{BASE_URL}/api/v1/tasks/{cancel_id}/cancel")
        assert r.status_code in (200, 202), f"Cancel failed: {r.status_code} {r.text}"
        print(f"PASS: POST /api/v1/tasks/{cancel_id}/cancel -> {r.json()}")

        print("\n==================================================")
        print("ALL LIVE END-TO-END FEATURE TESTS PASSED!")
        print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
