"""Short-term working memory with LangGraph checkpointed thread state."""

from __future__ import annotations

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.models import DemoResult, MemoryMessage


def sliding_window(messages: list[MemoryMessage], keep_last: int) -> list[MemoryMessage]:
    """Keep only the most recent messages."""

    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    return messages[-keep_last:]


def _message_role(message_type: str) -> str:
    if message_type == "human":
        return "user"
    if message_type == "ai":
        return "assistant"
    return message_type


def _messages_from_agent_state(state_messages: list[object]) -> list[MemoryMessage]:
    messages: list[MemoryMessage] = []
    for message in state_messages:
        role = _message_role(str(getattr(message, "type", "assistant")))
        content = getattr(message, "content", "")
        if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
            messages.append(MemoryMessage(role=role, content=content))
    return messages


def create_short_term_agent(settings: MemorySettings):
    """Create a LangGraph agent with thread-scoped short-term memory."""

    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key),
        tools=[],
        checkpointer=InMemorySaver(),
        prompt=(
            "You are a natural, direct chat assistant. Use the conversation history "
            "stored in this thread as short-term memory, especially to resolve "
            "follow-up questions like 'what about there?' or 'what about India?'. "
            "Answer the user's current question in 1-3 plain sentences. Do not "
            "mention memory, thread state, or offer extra follow-up tasks unless "
            "the user asks for them."
        ),
    )


def run_short_term_turn(
    agent: object,
    user_message: str,
    thread_id: str = "demo-short-term",
) -> DemoResult:
    """Invoke one turn of a checkpointed short-term-memory conversation."""

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        {"configurable": {"thread_id": thread_id}},
    )
    messages = _messages_from_agent_state(response["messages"])
    return DemoResult(
        name="Short-term working memory",
        technique="LangGraph checkpointed thread state",
        prompt_context=messages,
        output=messages[-1].content if messages else "",
        metadata={
            "thread_id": thread_id,
            "messages_in_thread": str(len(messages)),
        },
    )


def run_short_term_demo(
    settings: MemorySettings | None = None,
    thread_id: str = "demo-short-term",
) -> DemoResult:
    """Run a scripted multi-turn demo using LangGraph short-term memory."""

    settings = settings or MemorySettings.from_env()
    agent = create_short_term_agent(settings)
    run_short_term_turn(
        agent,
        "My name is Maya and I prefer concise explanations.",
        thread_id,
    )
    return run_short_term_turn(
        agent,
        "What should you remember about my name and response style?",
        thread_id,
    )
