"""Demo runner exports loaded lazily to keep unit tests lightweight."""

from __future__ import annotations

__all__ = [
    "run_fact_extraction_demo",
    "run_reflection_demo",
    "run_semantic_store_demo",
    "run_short_term_demo",
    "run_summarization_demo",
    "run_vector_recall_demo",
    "sliding_window",
]


def __getattr__(name: str):
    if name == "run_fact_extraction_demo":
        from ai_agents_memory.demos.fact_extraction import run_fact_extraction_demo

        return run_fact_extraction_demo
    if name == "run_reflection_demo":
        from ai_agents_memory.demos.reflection import run_reflection_demo

        return run_reflection_demo
    if name == "run_semantic_store_demo":
        from ai_agents_memory.demos.semantic_store import run_semantic_store_demo

        return run_semantic_store_demo
    if name in {"run_short_term_demo", "sliding_window"}:
        from ai_agents_memory.demos.short_term import run_short_term_demo, sliding_window

        return {"run_short_term_demo": run_short_term_demo, "sliding_window": sliding_window}[
            name
        ]
    if name == "run_summarization_demo":
        from ai_agents_memory.demos.summarization import run_summarization_demo

        return run_summarization_demo
    if name == "run_vector_recall_demo":
        from ai_agents_memory.demos.vector_recall import run_vector_recall_demo

        return run_vector_recall_demo
    raise AttributeError(name)
