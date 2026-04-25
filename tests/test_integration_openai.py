import os

import pytest

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.demos import (
    run_fact_extraction_demo,
    run_reflection_demo,
    run_semantic_store_demo,
    run_short_term_demo,
    run_summarization_demo,
    run_vector_recall_demo,
)

pytestmark = pytest.mark.requires_openai


def _settings_or_skip() -> MemorySettings:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")
    return MemorySettings.from_env()


def test_short_term_demo_calls_openai():
    result = run_short_term_demo(_settings_or_skip())

    assert result.output
    assert result.technique == "LangGraph checkpointed thread state"
    assert "Maya" in result.output or "concise" in result.output


def test_summarization_demo_calls_openai():
    result = run_summarization_demo(_settings_or_skip())

    assert result.output
    assert result.summary


def test_semantic_store_demo_calls_openai():
    result = run_semantic_store_demo(_settings_or_skip())

    assert result.output
    assert result.retrieved


def test_vector_recall_demo_calls_openai(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_SAMPLE_DATA_DIR", str(tmp_path))

    result = run_vector_recall_demo(_settings_or_skip())

    assert result.output
    assert result.retrieved


def test_fact_extraction_demo_calls_openai():
    result = run_fact_extraction_demo(_settings_or_skip())

    assert result.facts


def test_reflection_demo_calls_openai():
    result = run_reflection_demo(_settings_or_skip())

    assert result.reflections
