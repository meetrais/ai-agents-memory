from ai_agents_memory.models import DemoResult, ExtractedFact, RetrievedMemory
from ai_agents_memory.rendering import result_to_lines


def test_result_to_lines_includes_sections():
    result = DemoResult(
        name="Example",
        technique="Test technique",
        output="Done",
        facts=[
            ExtractedFact(
                subject="Maya",
                predicate="prefers",
                object="concise answers",
                confidence=0.9,
            )
        ],
        retrieved=[RetrievedMemory(content="Memory", score=0.4, source="test")],
    )

    lines = result_to_lines(result)

    assert lines[0] == "Example: Test technique"
    assert "Fact: Maya prefers concise answers. (0.90)" in lines
    assert "Retrieved[test score=0.400]: Memory" in lines
