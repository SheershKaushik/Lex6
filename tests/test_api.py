import json
from pathlib import Path

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


PASSCODE = "LEX6access404"


@pytest.fixture(autouse=True)
def installed_model(monkeypatch):
    problems = {"value": None}
    monkeypatch.setattr("lex6.api.app.check_model", lambda model, host: problems["value"])
    return problems


@pytest.fixture
def client(settings, fake_embedder):
    vs = VectorStore(
        fake_embedder.get_embedding_dim(), persist_directory=settings.vector_store_dir, embedding_model="fake"
    )
    pipeline = IngestionPipeline(fake_embedder, vs, FakeTranslator(), settings)
    llm = ScriptedLLM(fn=agent_llm)
    factories = []

    def factory(retriever, max_tokens, audience="expert"):
        factories.append((retriever, max_tokens, audience))
        return RetrievalAgent(retriever, llm, answer_max_tokens=max_tokens, audience=audience, score_threshold=-1.0)

    service = Lex6Service(vs, pipeline, RAGRetriever(vs, fake_embedder), factory, settings)
    c = TestClient(create_app(service))
    assert c.post("/api/login", json={"passcode": PASSCODE}).status_code == 200
    c.factories, c.llm, c.service = factories, llm, service
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
    retriever, max_tokens, _ = client.factories[-1]
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


def test_ollama_failure_returns_503_with_fix_and_health_reports_it(client, installed_model):
    from lex6.llm import LLMError

    def missing(prompt, system):
        raise LLMError("Ollama model 'deepseek-r1:8b' is not installed. Run `ollama pull deepseek-r1:8b`")

    client.llm.fn = missing
    r = client.post("/api/assessments", json={"query": "What about leakage?"})
    assert r.status_code == 503
    assert "ollama pull deepseek-r1:8b" in r.json()["detail"]
    assert client.get("/api/health").json()["llm"]["ok"] is True

    installed_model["value"] = "Ollama model 'deepseek-r1:8b' is not installed"
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["llm"] == {
        "model": client.get("/api/health").json()["llm_model"],
        "ok": False,
        "error": "Ollama model 'deepseek-r1:8b' is not installed",
    }


def test_corpus_indexed_once_listed_with_session_uploads(client, settings, fake_embedder, sample_dir):
    settings.corpus_dir = sample_dir
    first = client.service.sync_corpus()
    assert first["status"] == "ready" and first["files"] == first["ingested"] == 4 and not first["failed"]
    chunks = client.service.vector_store.ntotal

    vs = VectorStore(
        fake_embedder.get_embedding_dim(), persist_directory=settings.vector_store_dir, embedding_model="fake"
    )
    assert vs.ntotal == chunks
    restarted = Lex6Service(
        vs,
        IngestionPipeline(fake_embedder, vs, FakeTranslator(), settings),
        RAGRetriever(vs, fake_embedder),
        None,
        settings,
    )
    second = restarted.sync_corpus()
    assert second["ingested"] == 0 and second["already_indexed"] == 4 and vs.ntotal == chunks

    note = b"My own note: corresponding adjustments must be reported in the biennial transparency report."
    r = client.post("/api/documents", files=[("files", ("my_note.txt", note, "text/plain"))])
    assert r.json()["reports"][0]["status"] == "ingested"
    docs = {d["filename"]: (d["source"], d["session_upload"]) for d in client.get("/api/documents").json()}
    assert docs["Article6_glossary.txt"] == ("corpus", False)
    assert docs["my_note.txt"] == ("upload", True)
    assert len(docs) == 5
    assert client.get("/api/health").json()["corpus"]["status"] == "ready"


def test_server_startup_syncs_corpus_in_background(client, settings, sample_dir):
    import time

    settings.corpus_dir = sample_dir
    with TestClient(create_app(client.service)) as c:
        c.post("/api/login", json={"passcode": PASSCODE})
        deadline = time.time() + 60
        while c.get("/api/health").json()["corpus"]["status"] != "ready" and time.time() < deadline:
            time.sleep(0.1)
        assert c.get("/api/health").json()["corpus"]["ingested"] == 4
        assert len(c.get("/api/documents").json()) == 4


def test_missing_corpus_dir(client, settings, tmp_path):
    settings.corpus_dir = tmp_path / "nope"
    assert client.service.sync_corpus()["status"] == "missing"


def test_ask_answer_length_is_uncapped_by_default(client, sample_dir):
    upload(client, ("glossary.txt", (sample_dir / "Article6_glossary.txt").read_bytes(), "text/plain"))
    assert client.post("/api/assessments", json={"query": "What is leakage?"}).status_code == 200
    assert client.factories[-1][1:] == (None, "expert")
    answer_prompt = next(p for p in reversed(client.llm.prompts) if "### Context" in p["prompt"])
    assert answer_prompt["max_tokens"] is None and "There is no length" in answer_prompt["prompt"]

    r = client.post("/api/assessments", json={"query": "What is leakage?", "audience": "learner", "max_tokens": 65536})
    assert r.status_code == 200 and client.factories[-1][1:] == (65536, "learner")
    assert "student new to climate law" in client.llm.prompts[-1]["prompt"]
    assert client.post("/api/assessments", json={"query": "q", "audience": "child"}).status_code == 422

    html = client.get("/").text
    assert '<option value="expert" selected>Law Expert</option>' in html and "max_tokens" not in html


def test_passcode_login_gates_ui_and_api(client):
    anon = TestClient(client.app)
    r = anon.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert anon.get("/static/index.html", follow_redirects=False).status_code == 303
    assert anon.get("/api/documents").status_code == 401
    assert anon.post("/api/assessments", json={"query": "q"}).status_code == 401
    login_page = anon.get("/login")
    assert login_page.status_code == 200 and 'type="password"' in login_page.text
    assert "LEX6access404" not in login_page.text
    assert anon.get("/static/du.png").status_code == 200

    assert anon.post("/api/login", json={"passcode": "wrong"}).status_code == 401
    assert "lex6_access" not in anon.cookies
    assert anon.post("/api/login", json={"passcode": "lex6access404"}).status_code == 401
    assert anon.get("/api/documents").status_code == 401

    assert anon.post("/api/login", json={"passcode": PASSCODE}).status_code == 200
    assert anon.get("/").status_code == 200 and "Compliance Workspace" in anon.get("/").text
    assert anon.get("/api/documents").status_code == 200
    assert anon.get("/login", follow_redirects=False).status_code == 303

    assert anon.post("/api/logout").status_code == 200
    assert anon.get("/api/documents").status_code == 401

    forged = TestClient(client.app, cookies={"lex6_access": "forged"})
    assert forged.get("/api/documents").status_code == 401


def test_passcode_is_configurable_and_can_be_disabled(client, settings):
    import hashlib

    import lex6.config
    from lex6.api.app import passcode_matches

    assert PASSCODE not in Path(lex6.config.__file__).read_text()
    assert passcode_matches(PASSCODE, settings.access_passcode_sha256)

    settings.access_passcode_sha256 = hashlib.sha256(b"other").hexdigest()
    c = TestClient(create_app(client.service))
    assert c.post("/api/login", json={"passcode": PASSCODE}).status_code == 401
    assert c.post("/api/login", json={"passcode": "other"}).status_code == 200
    assert c.get("/api/documents").status_code == 200

    settings.access_passcode_sha256 = ""
    assert TestClient(create_app(client.service)).get("/api/documents").status_code == 200
