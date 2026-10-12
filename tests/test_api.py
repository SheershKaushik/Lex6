import json

import pytest
from fastapi.testclient import TestClient

from lex6.agent import RetrievalAgent
from lex6.api import Lex6Service, create_app
from lex6.api.app import DocumentFilterRetriever
from lex6.ingestion.pipeline import IngestionPipeline
from lex6.retrieval import RAGRetriever
from lex6.vectorstore import VectorStore
from tests.conftest import FakeTranslator, ScriptedLLM


def agent_llm(prompt, system):
    if "### Task: ROUTE" in prompt:
        return json.dumps({"retrieve": "hello" not in prompt.lower(), "reason": "test"})
    if "### Task: PLAN" in prompt:
        return json.dumps({"queries": []})
    if "### Task: REFLECT" in prompt:
        return json.dumps({"sufficient": True, "missing": "", "queries": []})
    if "### Context" in prompt:
        return "Leakage is deducted at ten percent [1]."
    return "I am Lex6."


@pytest.fixture
def client(settings, fake_embedder):
    vs = VectorStore(
        fake_embedder.get_embedding_dim(), persist_directory=settings.vector_store_dir, embedding_model="fake"
    )
    pipeline = IngestionPipeline(fake_embedder, vs, FakeTranslator(), settings)
    llm = ScriptedLLM(fn=agent_llm)
    factories = []

    def factory(retriever, max_tokens):
        factories.append((retriever, max_tokens))
        return RetrievalAgent(retriever, llm, answer_max_tokens=max_tokens, score_threshold=-1.0)

    service = Lex6Service(vs, pipeline, RAGRetriever(vs, fake_embedder), factory, settings)
    c = TestClient(create_app(service))
    c.factories, c.llm = factories, llm
    return c


def upload(client, *files):
    return client.post("/api/documents", files=[("files", f) for f in files])


def test_index_static_and_health(client):
    html = client.get("/").text
    assert "JustN0W ASPA-6" in html and ".docx" in html
    assert client.get("/static/du.png").headers["content-type"] == "image/png"
    assert client.get("/api/health").json()["status"] == "ok"


def test_upload_list_pages_and_chunks(client, sample_dir):
    txt = (sample_dir / "Article6_glossary.txt").read_bytes()
    r = upload(client, ("../../evil/glossary.txt", txt, "text/plain"), ("bad.pdf", b"not a pdf", "application/pdf"))
    body = r.json()
    assert r.status_code == 200 and len(body["document_ids"]) == 1
    assert [rep["status"] for rep in body["reports"]] == ["ingested", "rejected"]

    docs = client.get("/api/documents").json()
    assert [d["title"] for d in docs] == ["glossary.txt"]
    doc_id = docs[0]["id"]
    pages = client.get(f"/api/documents/{doc_id}/pages").json()
    assert pages[0]["page_number"] == 1 and "ITMO" in pages[0]["text"]
    chunks = client.get(f"/api/documents/{doc_id}/pages/1/clauses").json()
    assert chunks and chunks[0]["document_id"] == doc_id
    assert client.get(f"/api/clauses/{chunks[0]['id']}").json()["text"] == chunks[0]["text"]
    assert client.get("/api/documents/nope/pages").status_code == 404
    assert client.get("/api/clauses/nope").status_code == 404
    assert upload(client, ("again.txt", txt, "text/plain")).json()["reports"][0]["status"] == "rejected"


def test_ask_runs_agent_with_citations_and_filter(client, sample_dir):
    upload(client, ("glossary.txt", (sample_dir / "Article6_glossary.txt").read_bytes(), "text/plain"))
    doc_id = client.get("/api/documents").json()[0]["id"]

    r = client.post("/api/assessments", json={"query": "What is leakage?", "document_id": doc_id, "max_tokens": 1024})
    a = r.json()
    assert r.status_code == 200
    assert a["short_answer"] == "Leakage is deducted at ten percent [1]."
    assert a["used_retrieval"] and a["citations"][0]["source"] == "glossary.txt"
    assert a["detailed_explanation"][0].startswith("Cited sources:\n[1] glossary.txt")
    retriever, max_tokens = client.factories[-1]
    assert isinstance(retriever, DocumentFilterRetriever) and max_tokens == 1024

    direct = client.post("/api/assessments", json={"query": "Hello there"}).json()
    assert direct["used_retrieval"] is False and "no document retrieval" in direct["detailed_explanation"][0]
    assert not isinstance(client.factories[-1][0], DocumentFilterRetriever)

    listed = client.get("/api/assessments").json()
    assert [x["query"] for x in listed] == ["What is leakage?", "Hello there"]
    assert client.post("/api/assessments", json={"query": "x", "document_id": "nope"}).status_code == 404
    assert client.post("/api/assessments", json={"query": ""}).status_code == 422


def test_document_filter_retriever():
    class Base:
        def retrieve(self, query, top_k=5, score_threshold=0.0):
            self.top_k = top_k
            return [{"id": i, "metadata": {"document_id": "a" if i % 2 else "b"}} for i in range(top_k)]

    base = Base()
    out = DocumentFilterRetriever(base, "a").retrieve("q", top_k=2)
    assert base.top_k == 8 and [d["id"] for d in out] == [1, 3]
