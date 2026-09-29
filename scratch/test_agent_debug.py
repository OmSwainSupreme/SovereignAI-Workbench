import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from backend.app.api.v1.tasks_router import (
    get_model_router,
    get_tool_registry,
    get_tool_executor,
    get_workspace,
    get_planner,
    get_verifier,
    get_agent_config,
)
from backend.app.services.model_service import get_model_service
from core.agent import Agent

async def main():
    router = get_model_router()
    gateway = get_model_service().gateway
    workspace = get_workspace()
    registry = get_tool_registry()
    executor = get_tool_executor(registry, workspace)
    planner = get_planner()
    verifier = get_verifier()
    config = get_agent_config()

    agent = Agent(
        model_router=router,
        model_gateway=gateway,
        tool_executor=executor,
        planner=planner,
        verifier=verifier,
        config=config,
        tool_registry=registry,
    )

    t0 = time.time()
    task = "Explain abstraction, encapsulation, inheritance and polymorphism in C++ with examples."
    print(f"Starting agent run for task: {task}\n", flush=True)

    th_chunks = 0
    ct_chunks = 0

    async def on_event(ev):
        nonlocal th_chunks, ct_chunks
        t = time.time() - t0
        ev_type = ev.get("type")
        if ev_type == "thinking_token":
            th_chunks += 1
            if th_chunks == 1 or th_chunks % 30 == 0:
                print(f"[{t:5.1f}s] thinking #{th_chunks}: accum_len={len(ev.get('accumulated', ''))}", flush=True)
        elif ev_type == "token":
            ct_chunks += 1
            if ct_chunks == 1 or ct_chunks % 30 == 0:
                print(f"[{t:5.1f}s] content #{ct_chunks}: accum_len={len(ev.get('accumulated', ''))}", flush=True)
        else:
            print(f"[{t:5.1f}s] event: {ev_type} -> {ev}", flush=True)

    try:
        res = await agent.run(task, on_event=on_event)
        print(f"\nFinished! Status: {res.status}, Error: {res.error}", flush=True)
        print(f"Messages count: {len(res.messages)}")
        for idx, m in enumerate(res.messages):
            print(f"Msg {idx} [{m.role}]: thinking_len={len(getattr(m, 'thinking', '') or '')}, content_len={len(m.content)}")
            print(f"Content preview: {m.content[:200]}...\n")
    except Exception as exc:
        print(f"EXCEPTION in agent.run: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
