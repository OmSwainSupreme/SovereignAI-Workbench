import asyncio
import os
import sys

sys.path.insert(0, "C:\\sovereign-ai")
sys.path.insert(0, "C:\\sovereign-ai\\backend")

from backend.app.api.v1.tasks_router import run_agent_task, task_store
from core.security.policy_engine import init_policy_engine

async def main():
    init_policy_engine()
    
    print("--- Testing 2: Coding Question ---")
    task2_id = "test-coding-1"
    prompt2 = "Write a Python function to check if a string is a palindrome and test it with a few examples."
    
    await task_store.create_task_record(
        task_id=task2_id,
        task_description=prompt2,
        attachments=[],
        result=None,
        created_at=0,
    )
    
    try:
        await run_agent_task(task2_id, prompt2, attachments=[])
        rec2 = await task_store.get_task_record(task2_id)
        task2 = rec2.to_task()
        print(f"Task 2 Status: {task2.status}, Model: {task2.model}")
        for m in task2.messages:
            print(f"[{m.role}]:\n{m.content}\n")
    except Exception as e:
        print(f"Task 2 Error: {e}")
        import traceback
        traceback.print_exc()

    print("--- Testing 3: Vision Question ---")
    task3_id = "test-vision-1"
    prompt3 = "Answer question 2 from the attached image."
    attachment3 = "Question.png"
    
    await task_store.create_task_record(
        task_id=task3_id,
        task_description=prompt3,
        attachments=[attachment3],
        result=None,
        created_at=0,
    )
    
    try:
        await run_agent_task(task3_id, prompt3, attachments=[attachment3])
        rec3 = await task_store.get_task_record(task3_id)
        task3 = rec3.to_task()
        print(f"Task 3 Status: {task3.status}, Model: {task3.model}")
        print(f"Task 3 Tool Events: {[t.label + ': ' + t.status for t in task3.toolEvents]}")
        for m in task3.messages:
            print(f"[{m.role}]:\n{m.content}\n")
    except Exception as e:
        print(f"Task 3 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
