"""Console rendering helpers for CLI demos."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_agents_memory.models import DemoResult


def result_to_lines(result: DemoResult) -> list[str]:
    """Return a compact text representation useful for tests and logs."""

    lines = [f"{result.name}: {result.technique}", result.output]
    if result.summary:
        lines.append(f"Summary: {result.summary}")
    for fact in result.facts:
        lines.append(f"Fact: {fact.as_sentence()} ({fact.confidence:.2f})")
    for memory in result.retrieved:
        score = "" if memory.score is None else f" score={memory.score:.3f}"
        lines.append(f"Retrieved[{memory.source}{score}]: {memory.content}")
    for reflection in result.reflections:
        lines.append(f"Reflection: {reflection.lesson}")
    return lines


def render_result(console: Console, result: DemoResult) -> None:
    """Render a result with readable sections."""

    console.print(Panel.fit(result.output, title=f"{result.name} - {result.technique}"))
    if result.summary:
        console.print(Panel(result.summary, title="Summary"))
    if result.facts:
        table = Table(title="Extracted facts")
        table.add_column("Subject")
        table.add_column("Predicate")
        table.add_column("Object")
        table.add_column("Confidence", justify="right")
        for fact in result.facts:
            table.add_row(
                fact.subject,
                fact.predicate,
                fact.object,
                f"{fact.confidence:.2f}",
            )
        console.print(table)
    if result.retrieved:
        table = Table(title="Retrieved memories")
        table.add_column("Source")
        table.add_column("Score", justify="right")
        table.add_column("Content")
        for memory in result.retrieved:
            table.add_row(
                memory.source,
                "" if memory.score is None else f"{memory.score:.3f}",
                memory.content,
            )
        console.print(table)
    if result.reflections:
        table = Table(title="Reflections")
        table.add_column("Lesson")
        table.add_column("Evidence")
        table.add_column("Next action")
        for reflection in result.reflections:
            table.add_row(
                reflection.lesson,
                reflection.evidence,
                reflection.next_action,
            )
        console.print(table)
