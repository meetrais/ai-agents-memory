"""Structured fact extraction into typed semantic memories."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.fixtures import SAMPLE_DIALOGUE
from ai_agents_memory.models import DemoResult, ExtractedFact


class FactExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


def run_fact_extraction_demo(settings: MemorySettings | None = None) -> DemoResult:
    """Extract atomic user facts from a conversation."""

    from langchain_openai import ChatOpenAI

    settings = settings or MemorySettings.from_env()
    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    structured_llm = llm.with_structured_output(FactExtractionResult)
    extracted = structured_llm.invoke(
        [
            (
                "system",
                "Extract durable user facts as subject-predicate-object triples. "
                "Ignore transient small talk.",
            ),
            ("user", "\n".join(f"{m.role}: {m.content}" for m in SAMPLE_DIALOGUE)),
        ]
    )
    facts = extracted.facts
    return DemoResult(
        name="Structured fact extraction",
        technique="Atomic semantic facts",
        output=f"Extracted {len(facts)} durable facts.",
        facts=facts,
    )
