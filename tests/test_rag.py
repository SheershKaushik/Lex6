from lex6.rag import (
    NO_CONTEXT_ANSWER,
    AdvancedRAGPipeline,
    build_sources,
    format_context,
    rag_advanced,
    rag_simple,
)
from tests.conftest import ScriptedLLM

RESULTS = [
    {
        "id": "doc_a_0",
        "content": "Parties shall pursue domestic mitigation measures.",
        "metadata": {"source_file": "paris.pdf", "page": 3, "document_id": "d1"},
        "score": 0.8,
        "similarity_score": 0.8,
    },
    {
        "id": "doc_b_1",
        "content": "Les Parties ... " * 40,
        "metadata": {"source_file": "accord.pdf", "page": 0, "translated": True, "original_language": "fr"},
        "score": 3.2,
        "similarity_score": None,
    },
]


class StaticRetriever:
    def __init__(self, results):
        self.results, self.calls = results, []

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        self.calls.append((query, top_k, score_threshold))
        return self.results[:top_k]


def test_format_context_numbers_chunks_with_provenance():
    ctx = format_context(RESULTS)
    assert ctx.startswith("[1] Source: paris.pdf, page 4\nParties shall")
    assert "[2] Source: accord.pdf, page 1 (translated from fr)" in ctx


def test_build_sources():
    s = build_sources(RESULTS, preview_length=20)
    assert s[0] == {
        "index": 1,
        "chunk_id": "doc_a_0",
        "document_id": "d1",
        "source": "paris.pdf",
        "page": 4,
        "score": 0.8,
        "similarity_score": 0.8,
        "retrieval_source": None,
        "original_language": None,
        "translated": False,
        "preview": "Parties shall pursue...",
    }
    assert s[1]["translated"] is True


def test_rag_simple_prompt_and_no_context():
    llm = ScriptedLLM(["Answer [1]"])
    assert rag_simple("q?", StaticRetriever(RESULTS), llm, top_k=1) == "Answer [1]"
    prompt = llm.prompts[0]["prompt"]
    assert "[1] Source: paris.pdf" in prompt and "accord.pdf" not in prompt and "### Question\nq?" in prompt
    assert rag_simple("q?", StaticRetriever([]), ScriptedLLM()) == NO_CONTEXT_ANSWER


def test_rag_advanced_returns_sources_confidence_context():
    r = StaticRetriever(RESULTS)
    out = rag_advanced("q", r, ScriptedLLM(["A [1][2]"]), top_k=2, min_score=0.3, return_context=True)
    assert r.calls == [("q", 2, 0.3)]
    assert out["answer"] == "A [1][2]"
    assert out["confidence"] == 0.8
    assert [s["index"] for s in out["sources"]] == [1, 2]
    assert out["context"].startswith("[1]")
    empty = rag_advanced("q", StaticRetriever([]), ScriptedLLM(), return_context=True)
    assert empty == {"answer": NO_CONTEXT_ANSWER, "sources": [], "confidence": 0.0, "context": ""}


def test_pipeline_citations_summary_history():
    llm = ScriptedLLM(["Full answer [1]", "Short summary."])
    p = AdvancedRAGPipeline(StaticRetriever(RESULTS), llm)
    out = p.query("q", top_k=2, summarize=True)
    assert out["citations"] == ["[1] paris.pdf (page 4)", "[2] accord.pdf (page 1)"]
    assert out["summary"] == "Short summary."
    assert "Full answer [1]" in llm.prompts[1]["prompt"]
    p.query("q2", top_k=0)
    assert [h["question"] for h in p.history] == ["q", "q2"]
    assert p.history[1]["answer"] == NO_CONTEXT_ANSWER
