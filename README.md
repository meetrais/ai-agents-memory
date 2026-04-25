# ai-agents-memory

Runnable Python samples for LLM agent memory management techniques.

LLM agents are stateless between calls, so memory management is the craft of
choosing which facts, summaries, episodes, and tool outputs belong in the next
context window. This repo shows representative techniques using LangChain,
LangGraph, LlamaIndex, Chroma, and the OpenAI API.

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` or export the variables in your shell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-5.4-mini"
$env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

The demos require `OPENAI_API_KEY` and may incur API usage costs. Normal unit
tests avoid network calls; integration tests are marked `requires_openai` and
skip automatically when the key is missing.

## Run Samples

```powershell
memory-samples short-term
memory-samples summarize
memory-samples semantic-store
memory-samples vector-recall
memory-samples fact-extraction
memory-samples reflection
memory-samples all
```

## Technique Map

| Command | Technique | Duration | Cognitive function | Storage shape |
| --- | --- | --- | --- | --- |
| `short-term` | LangGraph checkpointed thread state | Short-term | Working memory | Thread messages |
| `summarize` | Running summary | Hybrid | Working + semantic | Summary text |
| `semantic-store` | Store-backed tool recall | Long-term | Preference/semantic | JSON document |
| `vector-recall` | Embeddings + semantic retrieval | Long-term | Episodic/semantic | Vector chunks |
| `fact-extraction` | Structured extraction | Long-term | Semantic | Atomic facts |
| `reflection` | Meta-cognitive consolidation | Long-term | Procedural/experiential | Lessons |

## Tests

```powershell
pytest
pytest -m requires_openai
```

## What Is Not Included Yet

This first pass keeps the examples concise and educational. It does not include
full MemGPT-style paging, dynamic MCP/tool eviction, subgoal working memory, or
production knowledge graph storage. Those are natural extension points once the
core sample shape is useful.

## Short-Term Chat

`memory-samples short-term` opens an interactive chat. The agent retains messages
inside the current process by using a LangGraph checkpointer and a `thread_id`.

```powershell
memory-samples short-term --thread-id demo
```

Type `quit`, `exit`, or `:q` to stop. `InMemorySaver` is intentionally used for
the sample; use a database-backed checkpointer when the same thread must survive
separate CLI runs or application restarts.
