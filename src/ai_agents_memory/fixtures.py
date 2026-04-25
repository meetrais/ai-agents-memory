"""Small deterministic inputs reused by the memory demos."""

from __future__ import annotations

from ai_agents_memory.models import MemoryMessage

SAMPLE_DIALOGUE: list[MemoryMessage] = [
    MemoryMessage(role="user", content="My name is Maya Chen."),
    MemoryMessage(role="assistant", content="Nice to meet you, Maya."),
    MemoryMessage(role="user", content="I run trail races on weekends."),
    MemoryMessage(role="assistant", content="Trail races sound like a great rhythm."),
    MemoryMessage(role="user", content="Please keep explanations concise."),
    MemoryMessage(role="assistant", content="Got it. I will keep answers concise."),
    MemoryMessage(role="user", content="I am planning a hydration strategy for a 25K."),
]

MEMORY_CORPUS: list[str] = [
    "Maya prefers concise explanations.",
    "Maya runs trail races on weekends.",
    "Maya is planning hydration for a 25K race.",
    "Jordan prefers examples in TypeScript.",
    "Ari is allergic to peanuts.",
]

FAILED_EPISODE = [
    "The agent recommended a long training plan without asking about race date.",
    "The user clarified that the race is in nine days.",
    "The answer improved after the agent focused on tapering and hydration.",
]
