import pytest
from core.agent.verifier import SimpleVerifier
from core.agent.planner import SimplePlanner
from core.agent.types import AgentMessage, AgentResult, AgentState, AgentStatus
from backend.app.api.v1.tasks_router import TaskRecord, TaskStore


@pytest.mark.asyncio
async def test_verifier_rejects_thinking_only():
    verifier = SimpleVerifier()
    state = AgentState(task_id="t1", task="Solve this")
    state.messages.append(AgentMessage(role="user", content="Solve this"))
    state.messages.append(AgentMessage(role="assistant", content="", thinking="I am thinking..."))
    
    result = await verifier.verify(state)
    assert result.passed is False
    assert "no final answer" in result.reason.lower()


@pytest.mark.asyncio
async def test_verifier_rejects_empty_content():
    verifier = SimpleVerifier()
    state = AgentState(task_id="t2", task="Solve this")
    state.messages.append(AgentMessage(role="user", content="Solve this"))
    state.messages.append(AgentMessage(role="assistant", content="   "))
    
    result = await verifier.verify(state)
    assert result.passed is False


@pytest.mark.asyncio
async def test_verifier_accepts_valid_content():
    verifier = SimpleVerifier()
    state = AgentState(task_id="t3", task="Solve this")
    state.messages.append(AgentMessage(role="user", content="Solve this"))
    state.messages.append(AgentMessage(role="assistant", content="Here is the complete solution.", thinking="Reasoning"))
    
    result = await verifier.verify(state)
    assert result.passed is True


@pytest.mark.asyncio
async def test_completion_invariant_enforces_failure_on_empty_content():
    store = TaskStore()
    state = AgentState(task_id="fail-task", task="A prompt")
    state.status = AgentStatus.COMPLETE  # Even if agent status was COMPLETE
    state.messages.append(AgentMessage(role="assistant", content="", thinking="Only thoughts"))
    result = AgentResult.from_state(state)

    has_assistant_content = any(
        getattr(m, "role", None) in ("assistant", "agent")
        and bool(getattr(m, "content", None) and str(getattr(m, "content", "")).strip())
        for m in (result.messages or [])
    )

    if result.status == AgentStatus.COMPLETE and has_assistant_content and not result.error:
        final_status = "completed"
        phase_label = "Completed"
    else:
        final_status = "failed"
        if not has_assistant_content:
            phase_label = "Generation interrupted: No final answer produced"
        else:
            phase_label = result.error or "Task failed"

    assert final_status == "failed"
    assert phase_label == "Generation interrupted: No final answer produced"


@pytest.mark.asyncio
async def test_completion_invariant_passes_with_valid_content():
    state = AgentState(task_id="pass-task", task="A prompt")
    state.status = AgentStatus.COMPLETE
    state.messages.append(AgentMessage(role="assistant", content="Valid final answer", thinking="Thoughts"))
    result = AgentResult.from_state(state)

    has_assistant_content = any(
        getattr(m, "role", None) in ("assistant", "agent")
        and bool(getattr(m, "content", None) and str(getattr(m, "content", "")).strip())
        for m in (result.messages or [])
    )

    if result.status == AgentStatus.COMPLETE and has_assistant_content and not result.error:
        final_status = "completed"
        phase_label = "Completed"
    else:
        final_status = "failed"
        phase_label = result.error or "Task failed"

    assert final_status == "completed"
    assert phase_label == "Completed"


@pytest.mark.asyncio
async def test_planner_routes_images_to_analyze_image():
    planner = SimplePlanner()
    available_tools = ["read_file", "analyze_image", "execute_code"]

    for ext in ["png", "jpg", "jpeg"]:
        img_name = f"question.{ext}"
        state = AgentState(task_id=f"img-{ext}", task="Read this image and solve", attachments=[img_name])
        plan = await planner.plan("Read this image and solve", available_tools, state)
        
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "analyze_image"
        assert plan.steps[0].inputs["path"] == img_name


@pytest.mark.asyncio
async def test_planner_routes_txt_to_read_file():
    planner = SimplePlanner()
    available_tools = ["read_file", "analyze_image", "execute_code"]

    state = AgentState(task_id="txt-task", task="Summarize this file", attachments=["sample.txt"])
    plan = await planner.plan("Summarize this file", available_tools, state)
    
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "read_file"
    assert plan.steps[0].inputs["path"] == "sample.txt"


@pytest.mark.asyncio
async def test_planner_does_not_mix_images_into_read_file():
    planner = SimplePlanner()
    available_tools = ["read_file", "analyze_image", "execute_code"]

    # When both are present (e.g. user attached txt then image)
    state = AgentState(
        task_id="mix-task",
        task="Analyze image and text",
        attachments=["sample.txt", "question.jpg"]
    )
    plan = await planner.plan("Analyze image and text", available_tools, state)
    
    # Image must be handled by vision, never read_file
    for step in plan.steps:
        if step.tool_name == "read_file":
            assert not step.inputs.get("path", "").lower().endswith((".jpg", ".png", ".jpeg"))
