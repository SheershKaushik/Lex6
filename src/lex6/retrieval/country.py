"""Country-priority rule on top of any retriever.

If the query names a country, chunks that mention it (or that come from a document whose filename country matches)
are put first with ``score = 1.0`` and ``country_match``; every matching document gets at least one chunk. The base
retriever's normal ranking follows.
"""

from __future__ import annotations

from typing import Any

from lex6.countries import canonical_country, extract_countries, mentions_country
from lex6.retrieval.base import Retriever, cosine_from_squared_l2, make_result, source_name
from lex6.vectorstore import VectorStore

COUNTRY_SCORE = 1.0


class CountryPriorityRetriever:
    def __init__(
        self, base: Retriever, vector_store: VectorStore, embedder: Any, max_country_chunks: int | None = None
    ):
        self.base, self.vector_store, self.embedder = base, vector_store, embedder
        self.max_country_chunks = max_country_chunks

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        results = self.base.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        countries = extract_countries(query)
        if not countries:
            return results
        matches = self._matching_chunks(query, set(countries))
        if not matches:
            return results
        selected = self._select(matches, self.max_country_chunks or top_k)
        selected_ids = {d["id"] for d in selected}
        selected += [
            {**matches[d["id"]], "similarity_score": d.get("similarity_score")}
            for d in results
            if d["id"] in matches and d["id"] not in selected_ids
        ]
        selected.sort(key=lambda d: d.get("similarity_score") or 0.0, reverse=True)
        ids = {d["id"] for d in selected}
        out = selected + [d for d in results if d["id"] not in ids]
        for rank, d in enumerate(out, 1):
            d["rank"] = rank
        return out

    def _matching_chunks(self, query: str, countries: set[str]) -> dict[str, dict[str, Any]]:
        vs = self.vector_store
        distances, indices = vs.search_embeddings(self.embedder.generate_embeddings([query]), vs.ntotal)
        cosine = {int(i): float(cosine_from_squared_l2(d)) for d, i in zip(distances, indices, strict=True) if i >= 0}
        matches: dict[str, dict[str, Any]] = {}
        for idx in range(vs.ntotal):
            chunk = vs.docstore.get_by_index(idx)
            if chunk is None:
                continue
            md = chunk["metadata"]
            hits = sorted(
                c for c in countries if canonical_country(md.get("country")) == c or mentions_country(chunk["text"], c)
            )
            if hits:
                matches[chunk["original_id"]] = make_result(
                    chunk,
                    score=COUNTRY_SCORE,
                    similarity_score=cosine.get(idx),
                    retrieval_source="country",
                    country_match=hits,
                )
        return matches

    @staticmethod
    def _select(matches: dict[str, dict[str, Any]], cap: int) -> list[dict[str, Any]]:
        """Round-robin over documents (best chunk of each first) so every matching document is represented."""
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for d in matches.values():
            by_doc.setdefault(d["metadata"].get("document_id") or source_name(d["metadata"]), []).append(d)
        queues = [sorted(g, key=lambda d: d.get("similarity_score") or 0.0, reverse=True) for g in by_doc.values()]
        cap = max(cap, len(queues))
        selected: list[dict[str, Any]] = []
        while len(selected) < cap and any(queues):
            for q in queues:
                if q and len(selected) < cap:
                    selected.append(q.pop(0))
        return selected
