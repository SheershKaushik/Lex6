"""Factories that wire the real components together from ``Settings``."""

from __future__ import annotations

from functools import lru_cache

from lex6.config import Settings, get_settings
from lex6.embeddings import EmbeddingManager
from lex6.ingestion.language import LLMTranslator
from lex6.ingestion.pipeline import IngestionPipeline
from lex6.llm import OllamaLLM
from lex6.vectorstore import VectorStore


@lru_cache
def get_embedder(model_name: str) -> EmbeddingManager:
    return EmbeddingManager(model_name)


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    s = settings or get_settings()
    embedder = get_embedder(s.embedding_model)
    return VectorStore(
        embedding_dim=embedder.get_embedding_dim(),
        collection_name=s.collection_name,
        persist_directory=s.vector_store_dir,
        embedding_model=s.embedding_model,
    )


def build_ingestion_pipeline(settings: Settings | None = None, vector_store: VectorStore | None = None):
    s = settings or get_settings()
    translator = LLMTranslator(OllamaLLM(model_name=s.effective_translation_model, temperature=0.0, settings=s))
    return IngestionPipeline(
        embedder=get_embedder(s.embedding_model),
        vector_store=vector_store or build_vector_store(s),
        translator=translator,
        settings=s,
    )
