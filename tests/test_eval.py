import pytest

from lex6.eval import (
    AgentSystem,
    EvalQuestion,
    NaiveBaseline,
    SystemOutput,
    format_report,
    load_questions,
    run_eval,
    score_output,
)
from lex6.eval.harness import is_abstention, summarize
from lex6.rag import INSUFFICIENT_CONTEXT

Q = EvalQuestion(
    id="q",
    question="How is leakage handled?",
    expected_sources=[["deal.docx"], ["paris.pdf", "glossary.txt"]],
    expected_keywords=[["ten percent", "10%"], ["leakage"]],
)


def src(name, i=1):
    return {"index": i, "source": name, "chunk_id": f"c{i}", "page": 1}


def test_builtin_question_set_is_valid():
    qs = load_questions()
    assert len(qs) >= 10
    assert len({q.id for q in qs}) == len(qs)
    assert {q.type for q in qs} == {"answerable", "unanswerable", "no_retrieval"}
    assert all(q.expected_sources and q.expected_keywords for q in qs if q.type == "answerable")


def test_score_answerable():
    out = SystemOutput(
        "A 10% deduction applies to leakage [1][2].",
        sources=[src("deal.docx", 1), src("other.pdf", 2)],
        citations=[src("deal.docx", 1), src("other.pdf", 2)],
        used_retrieval=True,
    )
    s = score_output(Q, out)
    assert s == {
        "routing": 1.0,
        "abstention": 1.0,
        "source_recall": 0.5,
        "citation_precision": 0.5,
        "keyword_coverage": 1.0,
    }


def test_score_no_citations_and_abstention():
    out = SystemOutput(INSUFFICIENT_CONTEXT, [src("deal.docx")], [], True)
    s = score_output(Q, out)
    assert s["citation_precision"] == 0.0 and s["abstention"] == 0.0
    unans = EvalQuestion(id="u", question="fine?", type="unanswerable")
    assert score_output(unans, out) == {"routing": 1.0, "abstention": 1.0}
    talk = EvalQuestion(id="t", question="hi", type="no_retrieval")
    assert score_output(talk, out) == {"routing": 0.0}
    assert is_abstention("No relevant context found to answer the question.")


class FakeAgent:
    def run(self, q):
        from lex6.agent import AgentResponse

        return AgentResponse(q, "Leakage: 10% [1].", True, [q], [src("deal.docx")], [src("deal.docx")])


class StaticRetriever:
    def retrieve(self, query, top_k=5, score_threshold=0.0):
        return [
            {
                "id": "c1",
                "content": "Leakage text",
                "metadata": {"source_file": "other.pdf", "page": 0},
                "score": 0.5,
                "similarity_score": 0.5,
            },
        ]


def test_systems_and_report():
    from tests.conftest import ScriptedLLM

    baseline = NaiveBaseline(StaticRetriever(), ScriptedLLM(fn=lambda p, s: "Leakage is handled [1] [4]."))
    out = baseline(Q.question)
    assert out.used_retrieval and [c["source"] for c in out.citations] == ["other.pdf"]

    def broken(q):
        raise ConnectionError("ollama down")

    rows = run_eval([Q], {"baseline": baseline, "agent": AgentSystem(FakeAgent()), "broken": broken})
    res = rows[0]["results"]
    assert res["agent"]["overall"] > res["baseline"]["overall"]
    assert res["broken"]["error"] == "ollama down" and res["broken"]["scores"]["routing"] == 0.0

    summary = summarize(rows, ["baseline", "agent"])
    assert summary["citation_precision"] == {"baseline": 0.0, "agent": 1.0}
    report = format_report(rows, ["baseline", "agent"])
    assert "Comparison" in report and "Δ (agent-baseline)" in report
    assert "| citation_precision" in report and "+1.000" in report


def test_report_handles_metric_missing_for_all_questions():
    talk = EvalQuestion(id="t", question="hi", type="no_retrieval")
    rows = run_eval(
        [talk], {"a": lambda q: SystemOutput("hi", [], [], False), "b": lambda q: SystemOutput("x", [], [], True)}
    )
    report = format_report(rows, ["a", "b"])
    assert "| source_recall" in report
    assert summarize(rows, ["a", "b"])["routing"] == pytest.approx({"a": 1.0, "b": 0.0})
