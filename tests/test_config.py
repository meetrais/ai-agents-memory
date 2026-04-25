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


def test_from_env_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        MemorySettings.from_env()
    except MissingOpenAIKeyError as exc:
        assert "OPENAI_API_KEY is required" in str(exc)
    else:
        raise AssertionError("Expected MissingOpenAIKeyError")


def test_from_env_loads_dotenv_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=dotenv-key\n"
        "OPENAI_MODEL=dotenv-model\n"
        "OPENAI_EMBEDDING_MODEL=dotenv-embedding\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)

    settings = MemorySettings.from_env()

    assert settings.openai_api_key == "dotenv-key"
    assert settings.openai_model == "dotenv-model"
    assert settings.openai_embedding_model == "dotenv-embedding"
