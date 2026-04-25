"""Typed result objects returned by the demo runners."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant", "tool"]


class MemoryMessage(BaseModel):
    role: MessageRole
    content: str


class RetrievedMemory(BaseModel):
    content: str
    score: float | None = None
    source: str


class ExtractedFact(BaseModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    def as_sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}."


class Reflection(BaseModel):
    lesson: str
    evidence: str
    next_action: str


class DemoResult(BaseModel):
    name: str
    technique: str
    prompt_context: list[MemoryMessage] = Field(default_factory=list)
    output: str
    summary: str | None = None
    facts: list[ExtractedFact] = Field(default_factory=list)
    retrieved: list[RetrievedMemory] = Field(default_factory=list)
    reflections: list[Reflection] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
