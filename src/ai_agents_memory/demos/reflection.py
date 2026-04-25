"""Reflection memory that distills episodes into reusable lessons."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.fixtures import FAILED_EPISODE
from ai_agents_memory.models import DemoResult, Reflection


class ReflectionResult(BaseModel):
    reflections: list[Reflection] = Field(default_factory=list)


def run_reflection_demo(settings: MemorySettings | None = None) -> DemoResult:
    """Ask an LLM to write higher-level lessons from an agent episode."""

    from langchain_openai import ChatOpenAI

    settings = settings or MemorySettings.from_env()
    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    structured_llm = llm.with_structured_output(ReflectionResult)
    result = structured_llm.invoke(
        [
            (
                "system",
                "Distill the episode into reusable agent lessons. Return concise "
                "reflections with evidence and a next action.",
            ),
            ("user", "\n".join(FAILED_EPISODE)),
        ]
    )
    return DemoResult(
        name="Reflection memory",
        technique="Meta-cognitive consolidation",
        output=f"Created {len(result.reflections)} reusable lessons.",
        reflections=result.reflections,
    )
