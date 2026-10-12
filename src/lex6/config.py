"""Central configuration. All values can be overridden with ``LEX6_*`` env vars or a ``.env`` file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEX6_", env_file=".env", extra="ignore")

    # LLMs (Ollama). Defaults mirror the reference notebook.
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "deepseek-r1:8b"
    llm_temperature: float = 0.1
    llm_num_predict: int = 8192
    # Ollama's default context window silently truncates long RAG prompts; raise it explicitly.
    llm_num_ctx: int = 8192
    translation_model: str | None = None
    hyde_model: str = "gpt-oss:120b-cloud"
    ollama_cloud_host: str = "https://ollama.com"

    # Embeddings / reranking (HuggingFace sentence-transformers).
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Validation.
    max_file_size_mb: float = 50.0
    min_text_chars: int = 20

    # Storage.
    data_dir: Path = Path("data")
    collection_name: str = "pdf_documents"

    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def effective_translation_model(self) -> str:
        return self.translation_model or self.llm_model

    @property
    def ollama_api_key(self) -> str | None:
        return os.getenv("OLLAMA_API_KEY") or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
