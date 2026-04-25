"""LlamaIndex vector memory / RAG recall with OpenAI embeddings and Chroma."""

from __future__ import annotations

from ai_agents_memory.config import MemorySettings
from ai_agents_memory.fixtures import MEMORY_CORPUS
from ai_agents_memory.models import DemoResult, RetrievedMemory


def run_vector_recall_demo(settings: MemorySettings | None = None) -> DemoResult:
    """Embed saved memories and retrieve the most relevant ones for a new query."""

    import chromadb
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.core.schema import Document
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

    settings = settings or MemorySettings.from_env()
    data_dir = settings.ensure_data_dir() / "chroma"
    client = chromadb.PersistentClient(path=str(data_dir))
    collection = client.get_or_create_collection("agent_memories")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = OpenAIEmbedding(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    Settings.llm = OpenAI(model=settings.openai_model, api_key=settings.openai_api_key)

    documents = [
        Document(text=text, metadata={"source": "memory_corpus"})
        for text in MEMORY_CORPUS
    ]
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    query = "What should the assistant remember when advising Maya about a race?"
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)

    return DemoResult(
        name="Vector recall",
        technique="LlamaIndex + Chroma semantic retrieval",
        output=str(response),
        retrieved=[
            RetrievedMemory(
                content=node.node.get_content(),
                score=node.score,
                source=str(node.node.metadata.get("source", "chroma")),
            )
            for node in nodes
        ],
    )
