"""Running-summary memory for older conversation turns."""

from __future__ import annotations

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.fixtures import SAMPLE_DIALOGUE
from ai_agents_memory.models import DemoResult, MemoryMessage


def run_summarization_demo(settings: MemorySettings | None = None) -> DemoResult:
    """Summarize older turns and answer using summary plus recent context."""

    from langchain_openai import ChatOpenAI

    settings = settings or MemorySettings.from_env()
    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    older = SAMPLE_DIALOGUE[:-2]
    recent = SAMPLE_DIALOGUE[-2:]
    summary_response = llm.invoke(
        [
            (
                "system",
                "Summarize durable user facts and preferences in two bullets or fewer.",
            ),
            ("user", "\n".join(f"{m.role}: {m.content}" for m in older)),
        ]
    )
    summary = str(summary_response.content)
    answer_response = llm.invoke(
        [
            ("system", f"Conversation summary:\n{summary}"),
            *[(message.role, message.content) for message in recent],
            ("user", "Give one concise planning tip that respects my preferences."),
        ]
    )
    return DemoResult(
        name="Summarization memory",
        technique="Running summary",
        prompt_context=[MemoryMessage(role="system", content=summary), *recent],
        output=str(answer_response.content),
        summary=summary,
    )
