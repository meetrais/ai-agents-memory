import pytest
from pydantic import ValidationError

from ai_agents_memory.models import ExtractedFact


def test_fact_sentence_formatting():
    fact = ExtractedFact(
        subject="Maya",
        predicate="prefers",
        object="concise answers",
        confidence=0.91,
    )

    assert fact.as_sentence() == "Maya prefers concise answers."


def test_fact_confidence_validation():
    with pytest.raises(ValidationError):
        ExtractedFact(
            subject="Maya",
            predicate="prefers",
            object="concise answers",
            confidence=1.2,
        )
