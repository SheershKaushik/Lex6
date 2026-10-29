"""Central configuration. All values can be overridden with ``LEX6_*`` env vars or a ``.env`` file."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """The Lex6 checkout that relative paths and ``.env`` are resolved against, whatever the working directory.

    ``LEX6_HOME`` if set, else the checkout this package was installed from (editable install), else the checkout
    holding the active virtualenv (``<repo>/.venv``), else the current directory.
    """
    if home := os.getenv("LEX6_HOME"):
        return Path(home).expanduser().resolve()
    for candidate in (Path(__file__).resolve().parents[2], Path(sys.prefix).resolve().parent):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lex6").is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEX6_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    # LLMs (Ollama). Defaults mirror the reference notebook.
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "deepseek-r1:8b"
    llm_temperature: float = 0.1
    # -1 = no cap on answer length: generation stops when the model finishes (or the context window is full).
    llm_num_predict: int = -1
    # Context window shared by the RAG prompt and the answer (and deepseek-r1's <think> reasoning). Ollama's
    # default silently truncates long prompts; raise it explicitly. Increase further if RAM/VRAM allows.
    llm_num_ctx: int = 16384
    # Extra tokens for capped calls (routing, planning, translation) on reasoning models such as deepseek-r1: their
    # <think> block counts toward num_predict, and Ollama's think=false does not stop deepseek-r1 reasoning.
    llm_reasoning_budget: int = 4096
    translation_model: str | None = None
    hyde_model: str = "gpt-oss:120b-cloud"
    ollama_cloud_host: str = "https://ollama.com"

    # Embeddings / reranking (HuggingFace sentence-transformers).
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Queries naming a country put chunks mentioning it first (score 1.0); the eval's naive baseline stays plain.
    country_priority: bool = True

    # Chunking.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Validation.
    max_file_size_mb: float = 50.0
    min_text_chars: int = 20

    # Storage. Relative paths are resolved against PROJECT_ROOT, not the directory `lex6` was launched from.
    data_dir: Path = Path("data")
    collection_name: str = "pdf_documents"
    # Corpus indexed into the persisted vector store when the server starts (recursively; vector_store/ and uploads/
    # are skipped); already-indexed files are skipped. Unset = the whole data_dir.
    corpus_dir: Path | None = None
    auto_ingest_corpus: bool = True

    # Web UI/API login. Only the SHA-256 of the passcode is stored; set LEX6_ACCESS_PASSCODE_SHA256 to change it
    # (`python -c "import hashlib; print(hashlib.sha256(b'NEW').hexdigest())"`). Empty string disables the login.
    access_passcode_sha256: str = "b611420fc0f90fc4d47d9ffe70a4249e5adf024b335af7f59efca34ccc3d8ae5"
    # Signs the login cookie. Unset = random per server start, so everyone logs in again after a restart.
    session_secret: str | None = None

    @model_validator(mode="after")
    def _anchor_paths(self) -> Settings:
        self.data_dir = _anchor(self.data_dir)
        self.corpus_dir = _anchor(self.corpus_dir) if self.corpus_dir else self.data_dir
        return self

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


def _anchor(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
