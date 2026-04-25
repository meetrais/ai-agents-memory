from ai_agents_memory.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    MemorySettings,
    MissingOpenAIKeyError,
)


def test_from_env_uses_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)

    settings = MemorySettings.from_env()

    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == DEFAULT_OPENAI_MODEL
    assert settings.openai_embedding_model == DEFAULT_EMBEDDING_MODEL


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        MemorySettings.from_env()
    except MissingOpenAIKeyError as exc:
        assert "OPENAI_API_KEY is required" in str(exc)
    else:
        raise AssertionError("Expected MissingOpenAIKeyError")
