import numpy as np
import pytest
from langchain_core.documents import Document

from lex6.retrieval import (
    AdvancedRAGRetriever,
    BM25SparseRetriever,
    RAGRetriever,
    cosine_from_squared_l2,
    page_label,
)
from lex6.retrieval.bm25 import tokenize
from lex6.vectorstore import VectorStore

TEXTS = [
    "Article 6 cooperative approaches allow internationally transferred mitigation outcomes (ITMOs).",
    "Each Party shall prepare nationally determined contributions every five years.",
    "Adaptation finance supports developing country Parties vulnerable to climate change.",
    "Corresponding adjustments avoid double counting of emission reductions under Article 6.",
    "STAN-METH-001 sets leakage requirements for removal activities.",
]


class KeywordCrossEncoder:
    """Scores (query, text) by token overlap, records calls."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return np.array([len(set(tokenize(q)) & set(tokenize(t))) for q, t in pairs], dtype="float32")


class FakeHyde:
    def __init__(self, text=None, fail=False):
        self.text, self.fail, self.prompts = text, fail, []

    def generate_text(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise ConnectionError("ollama down")
        return self.text


@pytest.fixture
def store(tmp_path, fake_embedder):
    vs = VectorStore(fake_embedder.dim, persist_directory=tmp_path, embedding_model="fake")
    docs = [Document(page_content=t, metadata={"source_file": f"f{i}.pdf", "page": i}) for i, t in enumerate(TEXTS)]
    vs.add_documents(docs, fake_embedder.generate_embeddings(TEXTS))
    return vs


def test_cosine_from_squared_l2():
    assert cosine_from_squared_l2(0.0) == pytest.approx(1.0)
    assert cosine_from_squared_l2(2.0) == pytest.approx(0.0)
    assert cosine_from_squared_l2(4.0) == pytest.approx(-1.0)


def test_page_label_is_one_based():
    assert page_label({"page": 0}) == 1
    assert page_label({}) is None


def test_dense_retriever_ranks_and_scores(store, fake_embedder):
    r = RAGRetriever(store, fake_embedder)
    res = r.retrieve(TEXTS[1], top_k=3)
    assert res[0]["content"] == TEXTS[1]
    assert res[0]["similarity_score"] == pytest.approx(1.0, abs=1e-5)
    assert [d["rank"] for d in res] == [1, 2, 3]
    assert all(res[i]["score"] >= res[i + 1]["score"] for i in range(len(res) - 1))
    assert res[0]["metadata"]["source_file"] == "f1.pdf"


def test_dense_retriever_threshold_and_empty_store(store, fake_embedder, tmp_path):
    r = RAGRetriever(store, fake_embedder)
    assert len(r.retrieve(TEXTS[0], top_k=5, score_threshold=0.99)) == 1
    empty = VectorStore(fake_embedder.dim, persist_directory=tmp_path / "e")
    assert RAGRetriever(empty, fake_embedder).retrieve("anything") == []


def test_bm25_strips_punctuation_and_drops_zero_scores(store):
    bm25 = BM25SparseRetriever(store.all_chunks())
    res = bm25.search("Article 6.", top_k=5)
    assert {d["text"] for d in res} == {TEXTS[0], TEXTS[3]}
    assert all(d["bm25_score"] > 0 for d in res)
    assert bm25.search("zzz unknown", top_k=5) == []
    assert BM25SparseRetriever([]).search("x") == []


def test_advanced_hyde_hybrid_rerank(store, fake_embedder):
    ce = KeywordCrossEncoder()
    hyde = FakeHyde(text="Leakage requirements for removal activities in STAN-METH-001.")
    r = AdvancedRAGRetriever(store, fake_embedder, hyde, ce)
    res = r.retrieve("STAN-METH-001 leakage", top_k=2)
    assert "STAN-METH-001" in hyde.prompts[0]
    assert res[0]["content"] == TEXTS[4]
    assert res[0]["retrieval_source"] == "hybrid"
    assert res[0]["cross_encoder_score"] == res[0]["score"]
    assert len(res) == 2 and [d["rank"] for d in res] == [1, 2]
    # the reranker sees the raw query, not the HyDE passage
    assert all(q == "STAN-METH-001 leakage" for q, _ in ce.pairs)


def test_advanced_falls_back_to_query_when_hyde_fails(store, fake_embedder):
    r = AdvancedRAGRetriever(store, fake_embedder, FakeHyde(fail=True), KeywordCrossEncoder())
    res = r.retrieve("nationally determined contributions", top_k=1)
    assert res[0]["content"] == TEXTS[1]


def test_advanced_threshold_and_sparse_only_candidates(store, fake_embedder):
    r = AdvancedRAGRetriever(store, fake_embedder, None, KeywordCrossEncoder(), fetch_multiplier=1)
    res = r.retrieve("double counting corresponding adjustments", top_k=5, score_threshold=2.0)
    assert [d["content"] for d in res] == [TEXTS[3]]
    sparse_only = [d for d in r.retrieve("ITMOs", top_k=5, score_threshold=-10) if d["retrieval_source"] == "sparse"]
    assert all(d["similarity_score"] is None for d in sparse_only)


def test_advanced_rebuilds_bm25_when_store_grows(store, fake_embedder):
    r = AdvancedRAGRetriever(store, fake_embedder, None, KeywordCrossEncoder())
    assert r.retrieve("glaciers", top_k=1, score_threshold=1) == []
    new = "Glaciers melting increases sea level."
    store.add_documents([Document(page_content=new, metadata={})], fake_embedder.generate_embeddings([new]))
    assert r.retrieve("glaciers", top_k=1, score_threshold=1)[0]["content"] == new


def test_advanced_without_cross_encoder_uses_dense_order(store, fake_embedder):
    r = AdvancedRAGRetriever(store, fake_embedder, None, None)
    res = r.retrieve(TEXTS[2], top_k=1)
    assert res[0]["content"] == TEXTS[2]
