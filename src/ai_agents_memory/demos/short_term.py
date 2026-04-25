"""Short-term working memory with sliding-window context."""

from __future__ import annotations

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.fixtures import SAMPLE_DIALOGUE
from ai_agents_memory.models import DemoResult, MemoryMessage


def sliding_window(messages: list[MemoryMessage], keep_last: int) -> list[MemoryMessage]:
    """Keep only the most recent messages."""

    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    return messages[-keep_last:]


def run_short_term_demo(
    settings: MemorySettings | None = None,
    keep_last: int = 4,
) -> DemoResult:
    """Run a small LLM call using only the last N messages as working memory."""

    from langchain_openai import ChatOpenAI

    settings = settings or MemorySettings.from_env()
    context = sliding_window(SAMPLE_DIALOGUE, keep_last)
    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    response = llm.invoke(
        [
            ("system", "Answer from only the visible short-term context."),
            *[(message.role, message.content) for message in context],
            ("user", "What should you remember about my response style?"),
        ]
    )
    return DemoResult(
        name="Short-term working memory",
        technique="Sliding window",
        prompt_context=context,
        output=str(response.content),
        metadata={"kept_messages": str(len(context))},
    )
