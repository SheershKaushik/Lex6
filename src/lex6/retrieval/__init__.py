"""Retrievers from the reference notebook: naive dense, BM25, cross-encoder and HyDE hybrid."""

from lex6.retrieval.advanced import AdvancedRAGRetriever
from lex6.retrieval.base import Retriever, cosine_from_squared_l2, make_result, page_label, source_name
from lex6.retrieval.bm25 import BM25SparseRetriever
from lex6.retrieval.country import CountryPriorityRetriever
from lex6.retrieval.dense import RAGRetriever
from lex6.retrieval.reranker import CrossEncoderWrapper

__all__ = [
    "AdvancedRAGRetriever",
    "BM25SparseRetriever",
    "CountryPriorityRetriever",
    "CrossEncoderWrapper",
    "RAGRetriever",
    "Retriever",
    "cosine_from_squared_l2",
    "make_result",
    "page_label",
    "source_name",
]
