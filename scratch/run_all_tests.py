import asyncio
import os
import sys

sys.path.insert(0, "C:\\sovereign-ai")
sys.path.insert(0, "C:\\sovereign-ai\\backend")

from backend.app.api.v1.tasks_router import run_agent_task, task_store
from core.security.policy_engine import init_policy_engine

async def main():
    init_policy_engine()
    
    # -------------------------------------------------------------
    # 1. Vision Model Question with Question.png
    # -------------------------------------------------------------
    print("==================================================")
    print("TEST 1: Vision Model Question (Question.png)")
    print("==================================================")
    task_vision_id = "test-vision-verify-1"
    prompt_vision = "Answer question 2 from the attached image."
    attachment = "Question.png"
    
    await task_store.create_task_record(
        task_id=task_vision_id,
        task_description=prompt_vision,
        attachments=[attachment],
        result=None,
        created_at=0,
    )
    
    try:
        await run_agent_task(task_vision_id, prompt_vision, attachments=[attachment])
        rec = await task_store.get_task_record(task_vision_id)
        task = rec.to_task()
        print(f"Status: {task.status}")
        print(f"Model: {task.model}")
        tool_event_strs = [f"{t.label} ({t.status})" for t in task.toolEvents]
        print(f"Tool Events: {tool_event_strs}")
        print("Assistant Message:")
        for m in task.messages:
            if m.role == "agent":
                print(m.content)
    except Exception as e:
        print(f"Vision Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------
    # 2. General Aptitude Question
    # -------------------------------------------------------------
    print("\n==================================================")
    print("TEST 2: General Aptitude Question")
    print("==================================================")
    task_aptitude_id = "test-aptitude-verify-1"
    prompt_aptitude = "A train 125 m long passes a man, running at 5 km/hr in the same direction in which the train is going, in 10 seconds. What is the speed of the train in km/hr?"
    
    await task_store.create_task_record(
        task_id=task_aptitude_id,
        task_description=prompt_aptitude,
        attachments=[],
        result=None,
        created_at=0,
    )
    
    try:
        await run_agent_task(task_aptitude_id, prompt_aptitude, attachments=[])
        rec = await task_store.get_task_record(task_aptitude_id)
        task = rec.to_task()
        print(f"Status: {task.status}")
        print(f"Model: {task.model}")
        print("Assistant Message:")
        for m in task.messages:
            if m.role == "agent":
                print(m.content)
    except Exception as e:
        print(f"Aptitude Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------
    # 3. Coding Question
    # -------------------------------------------------------------
    print("\n==================================================")
    print("TEST 3: Coding Question")
    print("==================================================")
    task_coding_id = "test-coding-verify-1"
    prompt_coding = "Create a Python function to reverse a singly linked list and explain its time and space complexity."
    
    await task_store.create_task_record(
        task_id=task_coding_id,
        task_description=prompt_coding,
        attachments=[],
        result=None,
        created_at=0,
    )
    
    try:
        await run_agent_task(task_coding_id, prompt_coding, attachments=[])
        rec = await task_store.get_task_record(task_coding_id)
        task = rec.to_task()
        print(f"Status: {task.status}")
        print(f"Model: {task.model}")
        print("Assistant Message:")
        for m in task.messages:
            if m.role == "agent":
                print(m.content)
    except Exception as e:
        print(f"Coding Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
