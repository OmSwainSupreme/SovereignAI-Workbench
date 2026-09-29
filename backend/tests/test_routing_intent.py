"""Tests for task intent detection and capability-based routing."""
import pytest
from core.agent.agent import Agent
from core.agent.types import AgentConfig, AgentState, Plan, PlanStep
from core.routing import (
    Capability,
    ModelRouter,
    RoutingRequest,
    TaskType,
)
from core.routing.registry import load_registry_from_path


@pytest.fixture
def router() -> ModelRouter:
    registry = load_registry_from_path("config/models.yaml")
    return ModelRouter(registry)


def test_routing_task_types(router: ModelRouter):
    """Verify that each TaskType maps to the correct specialized model."""
    # CHAT -> general (qwen3:4b)
    chat_dec = router.route(RoutingRequest(task_type=TaskType.CHAT))
    assert chat_dec.model.logical_name == "general"
    assert chat_dec.model.provider_model == "qwen3:4b"

    # DOCUMENT -> general (qwen3:4b)
    doc_dec = router.route(RoutingRequest(task_type=TaskType.DOCUMENT))
    assert doc_dec.model.logical_name == "general"
    assert doc_dec.model.provider_model == "qwen3:4b"

    # REASONING -> general (qwen3:4b)
    reas_dec = router.route(RoutingRequest(task_type=TaskType.REASONING))
    assert reas_dec.model.logical_name == "general"
    assert reas_dec.model.provider_model == "qwen3:4b"

    # CODING -> coding (qwen2.5-coder:3b)
    code_dec = router.route(RoutingRequest(task_type=TaskType.CODING))
    assert code_dec.model.logical_name == "coding"
    assert code_dec.model.provider_model == "qwen2.5-coder:3b"

    # VISION -> vision (qwen2.5vl:3b)
    vis_dec = router.route(RoutingRequest(task_type=TaskType.VISION))
    assert vis_dec.model.logical_name == "vision"
    assert vis_dec.model.provider_model == "qwen2.5vl:3b"


from unittest.mock import MagicMock


def test_agent_detect_task_type(router: ModelRouter):
    """Verify Agent._detect_task_type derives the proper intent from prompt/attachments/plan."""
    agent = Agent(
        model_router=router,
        model_gateway=MagicMock(),
        tool_executor=MagicMock(),
        planner=MagicMock(),
        verifier=MagicMock(),
    )

    # 1. Document summary
    state_doc = AgentState(task_id="t-1", task="Summarize this TXT file in 3 bullet points")
    assert agent._detect_task_type(state_doc) == TaskType.DOCUMENT

    # 2. Document attachment
    state_att = AgentState(task_id="t-2", task="What are the key points?", attachments=("notes.txt",))
    assert agent._detect_task_type(state_att) == TaskType.DOCUMENT

    # 3. Coding task
    state_code = AgentState(task_id="t-3", task="Write a C++ function to reverse a string")
    assert agent._detect_task_type(state_code) == TaskType.CODING

    # 4. Plan with execute_code
    state_plan_code = AgentState(
        task_id="t-4",
        task="Calculate primes",
        plan=Plan(goal="Calculate primes", steps=(PlanStep(step_id="1", description="run", tool_name="execute_code", inputs={}),))
    )
    assert agent._detect_task_type(state_plan_code) == TaskType.CODING

    # 5. Vision image attachment
    state_vis = AgentState(task_id="t-5", task="Describe what you see", attachments=("photo.png",))
    assert agent._detect_task_type(state_vis) == TaskType.VISION

    # 6. General explanation / chat
    state_chat = AgentState(task_id="t-6", task="Explain abstraction in C++ in simple words")
    assert agent._detect_task_type(state_chat) == TaskType.CHAT


def test_builtin_and_tasks_router_get_model_router():
    """Verify load_default_registry() and get_model_router() both route CHAT to general."""
    from core.routing.registry import load_default_registry
    from backend.app.api.v1.tasks_router import get_model_router

    # Builtin registry fallback
    builtin_router = ModelRouter(load_default_registry())
    dec1 = builtin_router.route(RoutingRequest(task_type=TaskType.CHAT))
    assert dec1.model.logical_name == "general"
    assert dec1.model.provider_model == "qwen3:4b"

    # Tasks router get_model_router()
    app_router = get_model_router()
    dec2 = app_router.route(RoutingRequest(task_type=TaskType.CHAT))
    assert dec2.model.logical_name == "general"
    assert dec2.model.provider_model == "qwen3:4b"


@pytest.mark.asyncio
async def test_planner_plans_vision_for_images():
    """Verify SimplePlanner plans analyze_image when an image is attached."""
    from core.agent.planner import SimplePlanner
    planner = SimplePlanner()
    state = AgentState(
        task_id="t-vis",
        task="Read this image and solve ALL the questions completely.",
        attachments=["assignment.png"],
    )
    plan = await planner.plan(
        task=state.task,
        available_tools=["list_files", "read_file", "write_file", "analyze_image", "ocr_image"],
        state=state,
    )
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "analyze_image"
    assert plan.steps[0].inputs["path"] == "assignment.png"
