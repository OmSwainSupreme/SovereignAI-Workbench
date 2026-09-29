import pytest
from core.agent.types import AgentMessage, AgentResult, AgentState, AgentStatus
from backend.app.api.v1.tasks_router import ChatMessage, TaskRecord, TaskStore


def test_agent_message_thinking_field():
    msg = AgentMessage(role="assistant", content="The final answer", thinking="Step 1: reasoning")
    assert msg.thinking == "Step 1: reasoning"
    assert msg.content == "The final answer"


def test_chat_message_thinking_dto():
    dto = ChatMessage(
        id="task-1-msg-0",
        role="agent",
        content="Final answer",
        createdAt="2026-09-12T00:00:00Z",
        thinking="Model thinking stream",
    )
    assert dto.thinking == "Model thinking stream"
    assert dto.content == "Final answer"


def test_task_record_preserves_thinking_in_to_task():
    state = AgentState(task_id="test-task", task="Solve problem")
    state.status = AgentStatus.COMPLETE
    state.messages.append(AgentMessage(role="user", content="Solve problem"))
    state.messages.append(AgentMessage(role="assistant", content="Answer", thinking="Real reasoning"))
    result = AgentResult.from_state(state)

    rec = TaskRecord(
        task_id="test-task",
        task_description="Solve problem",
        attachments=[],
        result=result,
        created_at=1000.0,
        completed_at=1010.0,
        status="completed",
        phase="completed",
        phase_label="Completed",
        model="qwen3:4b",
    )
    task = rec.to_task()
    assert len(task.messages) == 2
    assert task.messages[1].thinking == "Real reasoning"
    assert task.messages[1].content == "Answer"


def test_task_record_streaming_thinking_and_content():
    """Verify live pending message includes both thinking trace and content stream."""
    rec = TaskRecord(
        task_id="stream-task",
        task_description="Explain abstraction in C++",
        attachments=[],
        result=None,
        created_at=1000.0,
        status="running",
        phase="generating",
        phase_label="📝 Generating response...",
        model="qwen3:4b",
        streaming_thinking="I need to explain abstraction cleanly...",
        streaming_content="Abstraction in C++ is the concept of hiding details.",
    )
    task = rec.to_task()
    assert len(task.messages) == 2
    agent_msg = task.messages[1]
    assert agent_msg.role == "agent"
    assert agent_msg.pending is True
    assert agent_msg.thinking == "I need to explain abstraction cleanly..."
    assert agent_msg.content == "Abstraction in C++ is the concept of hiding details."


@pytest.mark.asyncio
async def test_task_store_thinking_to_content_transition():
    """Verify state transitions cleanly from thinking to token generation."""
    store = TaskStore()
    rec = await store.create_task_record(
        task_id="transition-task",
        task_description="Explain abstraction in C++",
        attachments=[],
        result=None,
        created_at=1000.0,
        status="running",
        phase="generating",
        phase_label="🧠 Thinking...",
    )
    # 1. While model is thinking
    await store.update_task_state(
        "transition-task",
        streaming_thinking="Step 1: Consider OOP.",
        phase="generating",
        phase_label="🧠 Thinking...",
    )
    r1 = await store.get_task_record("transition-task")
    assert r1.streaming_thinking == "Step 1: Consider OOP."
    assert r1.phase_label == "🧠 Thinking..."

    # 2. When first content token arrives
    await store.update_task_state(
        "transition-task",
        streaming_content="Abstraction is",
        phase="generating",
        phase_label="📝 Generating response...",
    )
    r2 = await store.get_task_record("transition-task")
    assert r2.streaming_thinking == "Step 1: Consider OOP."
    assert r2.streaming_content == "Abstraction is"
    assert r2.phase_label == "📝 Generating response..."
