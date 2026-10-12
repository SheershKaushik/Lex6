"""Sparse keyword retriever (notebook ``BM25SparseRetriever``)."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    # The notebook used ``str.split()``, so "Article 6." never matched "Article 6"; strip punctuation.
    return _TOKEN_RE.findall(text.lower())


class BM25SparseRetriever:
    """BM25Okapi over docstore records ``{"original_id", "text", "metadata"}``."""

    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in documents]) if documents else None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top:
            if scores[idx] > 0.0:
                doc = dict(self.documents[idx])
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)
        return results
