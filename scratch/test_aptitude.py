import asyncio
import os
import sys

# Ensure project root and backend are in python path
sys.path.insert(0, "C:\\sovereign-ai")
sys.path.insert(0, "C:\\sovereign-ai\\backend")

from backend.app.api.v1.tasks_router import run_agent_task, task_store
from core.security.policy_engine import init_policy_engine

async def main():
    init_policy_engine()
    print("Testing 1: General Aptitude Question...")
    task1_id = "test-aptitude-1"
    prompt1 = "A train travelling at 60 km/h crosses another train of length 150m coming from opposite direction in 9 seconds. If the speed of the second train is 40 km/h, what is the length of the first train in meters?"
    
    await task_store.create_task_record(
        task_id=task1_id,
        task_description=prompt1,
        attachments=[],
        result=None,
        created_at=0,
    )
    
    try:
        await run_agent_task(task1_id, prompt1, attachments=[])
        rec1 = await task_store.get_task_record(task1_id)
        task1 = rec1.to_task()
        print(f"Task 1 Status: {task1.status}, Model: {task1.model}")
        for m in task1.messages:
            print(f"[{m.role}]: {m.content[:200]}...")
    except Exception as e:
        print(f"Task 1 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
