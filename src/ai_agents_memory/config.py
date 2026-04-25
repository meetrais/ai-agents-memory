"""Runtime configuration for memory samples."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class MissingOpenAIKeyError(RuntimeError):
    """Raised when a sample needs OpenAI credentials that are not configured."""


class MemorySettings(BaseModel):
    """Settings shared by all demo runners."""

    openai_api_key: str = Field(repr=False)
    openai_model: str = DEFAULT_OPENAI_MODEL
    openai_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    data_dir: Path = Path(".memory_sample_data")

    @classmethod
    def from_env(cls, require_api_key: bool = True) -> "MemorySettings":
        """Create settings from environment variables."""

        api_key = os.getenv("OPENAI_API_KEY", "")
        if require_api_key and not api_key:
            raise MissingOpenAIKeyError(
                "OPENAI_API_KEY is required. Copy .env.example to .env or export "
                "OPENAI_API_KEY before running memory-samples."
            )

        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            data_dir=Path(os.getenv("MEMORY_SAMPLE_DATA_DIR", ".memory_sample_data")),
        )

    def ensure_data_dir(self) -> Path:
        """Create and return the local demo data directory."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir
