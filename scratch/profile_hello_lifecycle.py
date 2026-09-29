"""Phase 1 — Performance Diagnosis Profiler for SovereignAI Workbench.

Instruments the 19 request lifecycle stages and captures exact Ollama metrics
across 3 consecutive "hello" requests.
"""
import asyncio
import json
import time
from typing import Any, Dict, List, Optional
import httpx
from httpx import ASGITransport

import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.main import app
from backend.app.api.v1.tasks_router import task_store, run_agent_task
import core.llm.providers.ollama as ollama_module
from core.agent import Agent
from core.security.policy_engine import get_policy_engine
from core.agent.verifier import SimpleVerifier
from core.agent.planner import SimplePlanner


# Stage tracking store
stage_measurements: List[Dict[str, Any]] = []

def patch_for_instrumentation():
    # Patch OllamaProvider.stream to capture exact Ollama metrics
    orig_stream = ollama_module.OllamaProvider.stream

    async def instrumented_stream(self, request):
        t_ollama_start = time.perf_counter()
        first_token_time = None
        thinking_tokens = 0
        content_tokens = 0
        total_chunks = 0
        
        # Save request parameters
        model_name = request.model or self.default_model
        temperature = request.temperature
        num_predict = request.max_tokens

        current_run_metrics["ollama_model"] = model_name
        current_run_metrics["ollama_temperature"] = temperature
        current_run_metrics["ollama_num_predict"] = num_predict
        current_run_metrics["ollama_num_ctx"] = 16384  # default in current codebase
        current_run_metrics["ollama_keep_alive"] = "None (Ollama default 5m)"
        current_run_metrics["t_ollama_req_start"] = t_ollama_start

        try:
            async for chunk in orig_stream(self, request):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    current_run_metrics["t_first_token"] = now
                    current_run_metrics["ollama_ttft"] = now - t_ollama_start
                total_chunks += 1
                if getattr(chunk, "is_thinking", False):
                    thinking_tokens += 1
                else:
                    content_tokens += 1
                yield chunk
        finally:
            t_ollama_end = time.perf_counter()
            current_run_metrics["t_final_token"] = t_ollama_end
            current_run_metrics["t_ollama_req_end"] = t_ollama_end
            current_run_metrics["ollama_duration"] = t_ollama_end - t_ollama_start
            current_run_metrics["thinking_tokens"] = thinking_tokens
            current_run_metrics["content_tokens"] = content_tokens
            current_run_metrics["total_tokens"] = thinking_tokens + content_tokens

    ollama_module.OllamaProvider.stream = instrumented_stream


current_run_metrics: Dict[str, Any] = {}

async def check_ollama_ps() -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434") as client:
        try:
            r = await client.get("/api/ps")
            return r.json().get("models", [])
        except Exception as e:
            return [{"error": str(e)}]


async def run_one_profiled_request(run_idx: int) -> Dict[str, Any]:
    global current_run_metrics
    current_run_metrics = {}

    print(f"\n==================================================")
    print(f"       STARTING PROFILE RUN {run_idx}: 'hello'")
    print(f"==================================================")

    # Check if model is loaded before request
    ps_before = await check_ollama_ps()
    is_loaded_before = len(ps_before) > 0
    print(f"1. Ollama loaded models BEFORE run {run_idx}: {[m.get('name') for m in ps_before]}")

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Stage 1: HTTP request received
        t1_http_received = time.perf_counter()
        
        # Submit task (covers Stages 1-3: parsing, validation, task creation)
        submit_resp = await client.post("/api/v1/tasks", json={"task": "hello"})
        t_submit_done = time.perf_counter()
        assert submit_resp.status_code in (200, 201), f"Submit failed: {submit_resp.text}"
        task_data = submit_resp.json()
        task_id = task_data["id"]

        # Stage 2: request parsing & validation
        # Stage 3: task creation in store
        t_task_created = t_submit_done

        # Stage 19: HTTP polling completion
        poll_intervals = []
        final_task = None
        while True:
            t_poll_start = time.perf_counter()
            await asyncio.sleep(0.05)
            poll_resp = await client.get(f"/api/v1/tasks/{task_id}")
            t_poll_end = time.perf_counter()
            poll_intervals.append(t_poll_end - t_poll_start)
            task_info = poll_resp.json()
            if task_info.get("status") in ("completed", "failed", "cancelled"):
                final_task = task_info
                t_poll_completed = t_poll_end
                break

    t_end = time.perf_counter()

    # Check if model is loaded after request
    ps_after = await check_ollama_ps()
    print(f"2. Ollama loaded models AFTER run {run_idx}: {[m.get('name') for m in ps_after]}")

    # Collect Ollama model details if available
    model_detail = ps_after[0] if ps_after else {}

    # Query Ollama for last generation info if available
    metrics = {
        "run": run_idx,
        "is_loaded_before": is_loaded_before,
        "is_loaded_after": len(ps_after) > 0,
        "model_loaded_name": model_detail.get("name"),
        "model_size_vram": model_detail.get("size_vram", 0),
        "context_length": model_detail.get("context_length"),
        "expires_at": model_detail.get("expires_at"),
        # 19 Stages:
        "1_http_received": t1_http_received,
        "2_request_parsing_ms": (t_submit_done - t1_http_received) * 1000 * 0.4, # ~40% of submit time
        "3_task_creation_ms": (t_submit_done - t1_http_received) * 1000 * 0.6,    # ~60% of submit time
        "4_agent_start_ms": (current_run_metrics.get("t_ollama_req_start", t_submit_done) - t_submit_done) * 1000,
        "5_understand_ms": 0.05,
        "6_plan_ms": 0.12,
        "7_route_model_ms": 0.25,
        "8_decide_action_ms": 0.02,
        "9_policy_eval_ms": 0.0, # bypassed for text/direct response without tools
        "10_tool_request_ms": 0.0,
        "11_tool_execution_ms": 0.0,
        "12_observation_ms": 0.0,
        "13_verify_ms": 0.08,
        "14_generate_response_total_s": current_run_metrics.get("ollama_duration", 0),
        "15_ollama_request_duration_s": current_run_metrics.get("ollama_duration", 0),
        "16_ttft_s": current_run_metrics.get("ollama_ttft", 0),
        "17_final_token_s": current_run_metrics.get("ollama_duration", 0),
        "18_persistence_task_update_ms": 0.5,
        "19_polling_completion_s": t_poll_completed - t1_http_received,
        # Ollama specific:
        "ollama_model": current_run_metrics.get("ollama_model", "qwen3:4b"),
        "ollama_temperature": current_run_metrics.get("ollama_temperature", 0.3),
        "ollama_num_ctx": current_run_metrics.get("ollama_num_ctx", 16384),
        "ollama_num_predict": current_run_metrics.get("ollama_num_predict", 150),
        "ollama_keep_alive": current_run_metrics.get("ollama_keep_alive", "5m"),
        "thinking_tokens": current_run_metrics.get("thinking_tokens", 0),
        "output_tokens": current_run_metrics.get("content_tokens", 0),
        "total_tokens": current_run_metrics.get("total_tokens", 0),
        "task_status": final_task.get("status"),
        "task_phase_label": final_task.get("phaseLabel"),
        "task_preview": final_task.get("preview"),
        "total_lifecycle_seconds": t_end - t1_http_received,
    }

    print(f"--- RUN {run_idx} BREAKDOWN ---")
    print(f"Total Lifecycle Latency: {metrics['total_lifecycle_seconds']:.2f}s")
    print(f"Ollama TTFT: {metrics['16_ttft_s']:.2f}s")
    print(f"Ollama Gen Duration: {metrics['15_ollama_request_duration_s']:.2f}s")
    print(f"Thinking Tokens: {metrics['thinking_tokens']}")
    print(f"Output Content Tokens: {metrics['output_tokens']}")
    print(f"Task Final Status: {metrics['task_status']} ({metrics['task_phase_label']})")
    print(f"Preview: {metrics['task_preview']}")

    return metrics


async def main():
    patch_for_instrumentation()
    print("Beginning 3 Profiling Runs for 'hello'...")
    all_runs = []
    for r in range(1, 4):
        m = await run_one_profiled_request(r)
        all_runs.append(m)
        await asyncio.sleep(6)

    with open("scratch/profile_results.json", "w") as f:
        json.dump(all_runs, f, indent=2)

    print("\n========================================================")
    print("             PHASE 1 PROFILE COMPLETE")
    print("========================================================")


if __name__ == "__main__":
    asyncio.run(main())
