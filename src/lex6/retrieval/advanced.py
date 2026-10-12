"""HyDE -> hybrid (dense + BM25) -> cross-encoder reranking (notebook ``AdvancedRAGRetriever``)."""

from __future__ import annotations

import logging
from typing import Any

from lex6.retrieval.base import cosine_from_squared_l2, make_result
from lex6.retrieval.bm25 import BM25SparseRetriever
from lex6.vectorstore import VectorStore

logger = logging.getLogger(__name__)

HYDE_PROMPT = "Please write a brief, authoritative passage that perfectly answers the following query: {query}"


class AdvancedRAGRetriever:
    """Multi-stage retriever.

    1. HyDE: an LLM writes a hypothetical answer, which is embedded instead of the raw query.
    2. Hybrid search: dense FAISS search on the HyDE embedding + BM25 on the raw query (``fetch_k = 3*top_k``).
    3. The cross-encoder scores (raw query, chunk) pairs; results are sorted by that logit.

    The BM25 index is built from the vector store's docstore and rebuilt when the store grows.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: Any,
        llm_client: Any | None,
        cross_encoder: Any | None,
        sparse_retriever: BM25SparseRetriever | None = None,
        fetch_multiplier: int = 3,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.cross_encoder = cross_encoder
        self._sparse = sparse_retriever
        self._sparse_size = vector_store.ntotal if sparse_retriever is not None else -1
        self.fetch_multiplier = fetch_multiplier

    @property
    def sparse_retriever(self) -> BM25SparseRetriever:
        if self._sparse is None or self._sparse_size != self.vector_store.ntotal:
            self._sparse = BM25SparseRetriever(self.vector_store.all_chunks())
            self._sparse_size = self.vector_store.ntotal
        return self._sparse

    def hypothetical_document(self, query: str) -> str:
        if self.llm_client is None:
            return query
        try:
            text = self.llm_client.generate_text(HYDE_PROMPT.format(query=query))
        except Exception as e:  # HyDE is an optimisation; the raw query is a valid fallback.
            logger.warning("HyDE generation failed, using the raw query: %s", e)
            return query
        return text.strip() or query

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = -10.0, use_hyde: bool = True
    ) -> list[dict[str, Any]]:
        fetch_k = max(top_k * self.fetch_multiplier, top_k)
        hyde_text = self.hypothetical_document(query) if use_hyde else query

        candidates: dict[str, dict[str, Any]] = {}
        query_embedding = self.embedding_manager.generate_embeddings([hyde_text])
        distances, indices = self.vector_store.search_embeddings(query_embedding, fetch_k)
        for index, distance in zip(indices, distances, strict=True):
            if index == -1:
                continue
            doc_data = self.vector_store.docstore.get_by_index(index)
            if doc_data:
                candidates[doc_data["original_id"]] = make_result(
                    doc_data,
                    similarity_score=float(cosine_from_squared_l2(distance)),
                    bm25_score=None,
                    retrieval_source="dense",
                )

        for doc in self.sparse_retriever.search(query, fetch_k):
            existing = candidates.get(doc["original_id"])
            if existing is None:
                candidates[doc["original_id"]] = make_result(
                    doc, similarity_score=None, bm25_score=doc["bm25_score"], retrieval_source="sparse"
                )
            else:
                existing["bm25_score"] = doc["bm25_score"]
                existing["retrieval_source"] = "hybrid"

        candidate_list = list(candidates.values())
        if not candidate_list:
            return []

        if self.cross_encoder is None:
            logger.warning("No cross-encoder configured; returning unreranked hybrid candidates")
            ranked = sorted(candidate_list, key=lambda d: d["similarity_score"] or 0.0, reverse=True)
            for d in ranked:
                d["score"] = d["similarity_score"] or 0.0
        else:
            scores = self.cross_encoder.predict([[query, d["content"]] for d in candidate_list])
            for d, s in zip(candidate_list, scores, strict=True):
                d["cross_encoder_score"] = d["score"] = float(s)
            ranked = sorted(candidate_list, key=lambda d: d["score"], reverse=True)

        final = []
        for d in ranked:
            if d["score"] < score_threshold:
                continue
            d["rank"] = len(final) + 1
            final.append(d)
            if len(final) == top_k:
                break
        logger.info("Advanced retrieval: %d candidates -> %d results for %r", len(candidate_list), len(final), query)
        return final
