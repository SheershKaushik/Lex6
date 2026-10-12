"""Shared result schema for all retrievers.

Every retriever returns a list of dicts::

    {"id", "content", "metadata", "score", "similarity_score", "rank", "retrieval_source", ...}

``score`` is the value results are ranked by (cosine for dense, cross-encoder logit for advanced);
``similarity_score`` is always the dense cosine similarity (``None`` if the chunk came only from BM25).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = ...) -> list[dict[str, Any]]: ...


def cosine_from_squared_l2(distance: float | np.ndarray) -> float | np.ndarray:
    """Cosine similarity of unit vectors from FAISS ``IndexFlatL2`` (squared L2) distance."""
    return np.clip(1.0 - np.asarray(distance, dtype="float64") / 2.0, -1.0, 1.0)


def make_result(doc_data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "id": doc_data["original_id"],
        "content": doc_data["text"],
        "metadata": doc_data["metadata"],
        **extra,
    }


def source_name(metadata: dict[str, Any]) -> str:
    return str(metadata.get("source_file") or metadata.get("source") or "unknown")


def page_label(metadata: dict[str, Any]) -> int | None:
    """1-based page number (loaders store PyMuPDF's 0-based ``page``)."""
    page = metadata.get("page")
    return int(page) + 1 if isinstance(page, (int, np.integer)) else None
