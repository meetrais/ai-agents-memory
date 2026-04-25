"""LangGraph long-term semantic/preference memory using a store-backed tool."""

from __future__ import annotations

from typing import Annotated

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.models import DemoResult, RetrievedMemory


def run_semantic_store_demo(settings: MemorySettings | None = None) -> DemoResult:
    """Store a user preference in a LangGraph store and recall it with a tool."""

    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph.store.memory import InMemoryStore

    settings = settings or MemorySettings.from_env()
    store = InMemoryStore()
    namespace = ("users", "maya")
    store.put(namespace, "response_style", {"memory": "Maya prefers concise answers."})

    @tool
    def lookup_user_memory(key: Annotated[str, "Memory key to look up"]) -> str:
        """Look up a saved user memory by key."""

        item = store.get(namespace, key)
        if item is None:
            return "No memory found."
        return str(item.value["memory"])

    agent = create_react_agent(
        ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key),
        tools=[lookup_user_memory],
    )
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use lookup_user_memory with key response_style, then answer: "
                        "How should you format training advice for Maya?"
                    )
                )
            ]
        }
    )
    final_message = response["messages"][-1]
    return DemoResult(
        name="Semantic preference memory",
        technique="LangGraph store-backed tool",
        output=str(final_message.content),
        retrieved=[
            RetrievedMemory(
                content="Maya prefers concise answers.",
                source="langgraph.InMemoryStore",
            )
        ],
    )
