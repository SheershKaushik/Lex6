"""Naive single-shot dense retriever (notebook ``RAGRetriever``)."""

from __future__ import annotations

import logging
from typing import Any

from lex6.retrieval.base import cosine_from_squared_l2, make_result
from lex6.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Embed the query, search FAISS, convert distances to cosine similarity, apply a threshold."""

    def __init__(self, vector_store: VectorStore, embedding_manager: Any):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])
        distances, indices = self.vector_store.search_embeddings(query_embedding, top_k)
        results = []
        for index, distance in zip(indices, distances, strict=True):
            if index == -1:
                continue
            doc_data = self.vector_store.docstore.get_by_index(index)
            if not doc_data:
                continue
            similarity = float(cosine_from_squared_l2(distance))
            # float32 FAISS distances of orthogonal unit vectors can land a hair above 2.0 (cosine -1e-7)
            if similarity < score_threshold - 1e-6:
                continue
            results.append(
                make_result(
                    doc_data,
                    score=similarity,
                    similarity_score=similarity,
                    distance=float(distance),
                    rank=len(results) + 1,
                    retrieval_source="dense",
                )
            )
        logger.info("Dense retrieval: %d results for %r (threshold %.2f)", len(results), query, score_threshold)
        return results
