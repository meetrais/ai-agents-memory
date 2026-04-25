import pytest

from ai_agents_memory.demos.short_term import sliding_window
from ai_agents_memory.models import MemoryMessage


def test_sliding_window_keeps_recent_messages():
    messages = [
        MemoryMessage(role="user", content="one"),
        MemoryMessage(role="assistant", content="two"),
        MemoryMessage(role="user", content="three"),
    ]

    kept = sliding_window(messages, keep_last=2)

    assert [message.content for message in kept] == ["two", "three"]


def test_sliding_window_rejects_empty_window():
    with pytest.raises(ValueError, match="keep_last"):
        sliding_window([], keep_last=0)
