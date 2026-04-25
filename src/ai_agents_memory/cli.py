"""Command line interface for the memory samples."""

from __future__ import annotations

from collections.abc import Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ai_agents_memory.config import MemorySettings, MissingOpenAIKeyError
from ai_agents_memory.demos import (
    run_fact_extraction_demo,
    run_reflection_demo,
    run_semantic_store_demo,
    run_short_term_demo,
    run_summarization_demo,
    run_vector_recall_demo,
)
from ai_agents_memory.models import DemoResult
from ai_agents_memory.rendering import render_result

app = typer.Typer(help="Run LLM agent memory management code samples.")
console = Console()

DemoRunner = Callable[[MemorySettings], DemoResult]


def _settings() -> MemorySettings:
    try:
        return MemorySettings.from_env()
    except MissingOpenAIKeyError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run(runner: DemoRunner) -> None:
    render_result(console, runner(_settings()))


@app.command("short-term")
def short_term(
    thread_id: str = typer.Option(
        "demo-short-term",
        "--thread-id",
        "-t",
        help="Conversation thread id used by the LangGraph checkpointer.",
    ),
) -> None:
    """Chat with checkpointed short-term memory until you quit."""

    from ai_agents_memory.demos import create_short_term_agent, run_short_term_turn

    settings = _settings()
    agent = create_short_term_agent(settings)
    console.print(
        "[bold]Short-term memory chat[/bold] "
        f"(thread: [cyan]{thread_id}[/cyan]). Type [cyan]quit[/cyan] to exit."
    )

    while True:
        try:
            user_message = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_message:
            continue
        if user_message.lower() in {"quit", "exit", ":q"}:
            break

        result = run_short_term_turn(agent, user_message, thread_id)
        console.print(Panel(result.output, title="Agent"))


@app.command("summarize")
def summarize() -> None:
    """Run running-summary memory."""

    _run(run_summarization_demo)


@app.command("semantic-store")
def semantic_store() -> None:
    """Run LangGraph store-backed semantic memory."""

    _run(run_semantic_store_demo)


@app.command("vector-recall")
def vector_recall() -> None:
    """Run LlamaIndex vector memory recall."""

    _run(run_vector_recall_demo)


@app.command("fact-extraction")
def fact_extraction() -> None:
    """Run structured fact extraction."""

    _run(run_fact_extraction_demo)


@app.command("reflection")
def reflection() -> None:
    """Run reflection memory."""

    _run(run_reflection_demo)


@app.command("all")
def all_demos() -> None:
    """Run every memory sample."""

    settings = _settings()
    runners: list[DemoRunner] = [
        run_short_term_demo,
        run_summarization_demo,
        run_semantic_store_demo,
        run_vector_recall_demo,
        run_fact_extraction_demo,
        run_reflection_demo,
    ]
    for runner in runners:
        render_result(console, runner(settings))
