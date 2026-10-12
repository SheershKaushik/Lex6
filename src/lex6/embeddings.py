"""Embedding generation with sentence-transformers (BAAI/bge-m3 by default)."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str

    def get_embedding_dim(self) -> int: ...

    def generate_embeddings(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self._embedding_dim: int | None = None
        self._load_model()

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %s", self._embedding_dim)

    def get_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise ValueError("Model not loaded")
        return self._embedding_dim

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings with shape ``(len(texts), dim)``."""
        if self.model is None:
            raise ValueError("Model not loaded")
        if not texts:
            return np.zeros((0, self.get_embedding_dim()), dtype="float32")
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")
