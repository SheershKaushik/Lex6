import numpy as np
from langchain_core.documents import Document

from lex6.agent import RetrievalAgent
from lex6.countries import canonical_country, extract_countries, mentions_country
from lex6.retrieval import CountryPriorityRetriever, RAGRetriever
from lex6.vectorstore import VectorStore
from tests.conftest import ScriptedLLM


def test_extract_countries():
    assert extract_countries("How does the NDC of Australia mention climate law?") == ["Australia"]
    assert extract_countries("papua new guinea forests") == ["Papua New Guinea"]
    assert extract_countries("Nigeria's NDC") == ["Nigeria"]
    assert set(extract_countries("Compare the US and UK pledges")) == {"United States", "United Kingdom"}
    assert extract_countries("Tell us about leakage; turkey and chad are words too") == []
    assert set(extract_countries("Turkey and Russia")) == {"Russian Federation", "Türkiye"}
    assert extract_countries("South Korea") == ["South Korea"]
    assert canonical_country("australia") == "Australia" and canonical_country(None) is None
    assert mentions_country("Australia's Climate Change Act", "Australia")
    assert not mentions_country("Papua New Guinea", "Guinea")


DOCS = [
    (
        "au_ndc",
        "au_ndc.pdf",
        None,
        "Australia NDC: Australia commits to net zero by 2050 under its Climate Change Act.",
    ),
    ("au_ndc", "au_ndc.pdf", None, "The Australian Government reports emissions every year to parliament."),
    ("br_ndc", "br_ndc.pdf", None, "Brazil NDC mentions forests and climate law."),
    ("law", "law.txt", None, "Climate law frameworks: how does the NDC mention climate law and climate law reporting."),
    ("pac", "pacific.txt", None, "Pacific report: Australia and Fiji cooperate on adaptation."),
    ("strat", "Australia_strategy.txt", "Australia", "The strategy covers energy efficiency."),
]
QUERY = "How does the NDC of Australia mention climate law?"


def _store(fake_embedder, tmp_path):
    vs = VectorStore(fake_embedder.get_embedding_dim(), persist_directory=tmp_path / "vs", embedding_model="fake")
    docs = [
        Document(page_content=t, metadata={"document_id": d, "source_file": f, "country": c}) for d, f, c, t in DOCS
    ]
    vs.add_documents(docs, fake_embedder.generate_embeddings([d.page_content for d in docs]))
    return vs


def test_country_chunks_first_with_score_one(fake_embedder, tmp_path):
    vs = _store(fake_embedder, tmp_path)
    base = RAGRetriever(vs, fake_embedder)
    results = CountryPriorityRetriever(base, vs, fake_embedder).retrieve(QUERY, top_k=2, score_threshold=-1.0)
    country = [r for r in results if r.get("country_match")]
    assert {r["metadata"]["document_id"] for r in country} == {"au_ndc", "pac", "strat"}
    assert results[: len(country)] == country
    assert all(r["score"] == 1.0 and r["country_match"] == ["Australia"] for r in country)
    rest = results[len(country) :]
    assert rest and "law" in {r["metadata"]["document_id"] for r in rest}
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    assert all("Brazil" not in r["content"] for r in country)

    plain = CountryPriorityRetriever(base, vs, fake_embedder).retrieve("climate law reporting", top_k=2)
    assert plain == base.retrieve("climate law reporting", top_k=2)


def test_agent_keeps_country_chunks_above_higher_logits():
    class Stub:
        def retrieve(self, query, top_k=5, score_threshold=0.0):
            docs = [{"id": f"x{i}", "content": "x", "metadata": {}, "score": 9.0 - i} for i in range(9)]
            return docs + [
                {"id": "au", "content": "Australia", "metadata": {}, "score": 1.0, "country_match": ["Australia"]}
            ]

    agent = RetrievalAgent(Stub(), ScriptedLLM(), max_context_chunks=3)
    pool = {}
    agent._search([QUERY, "other"], pool)
    assert [d["id"] for d in agent._context(pool)] == ["au", "x0", "x1"]
    assert np.isclose(pool["au"]["score"], 1.0)
