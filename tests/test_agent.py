import json

import pytest

from lex6.agent import (
    RetrievalAgent,
    extract_citations,
    infer_citations,
    parse_json_object,
    sanitize_citations,
    verify_citations,
)
from lex6.rag import INSUFFICIENT_CONTEXT
from tests.conftest import ScriptedLLM


def chunk(i, text, score):
    return {
        "id": f"c{i}",
        "content": text,
        "metadata": {"source_file": f"doc{i}.pdf", "page": i, "document_id": f"d{i}"},
        "score": score,
        "similarity_score": score / 10,
    }


CORPUS = {
    "leakage": [chunk(1, "Leakage shall be monitored under STAN-METH-001.", 5.0)],
    "double counting": [chunk(2, "Corresponding adjustments avoid double counting.", 6.0)],
    "corresponding adjustments": [
        chunk(2, "Corresponding adjustments avoid double counting.", 6.5),
        chunk(3, "Adjustments are reported in biennial transparency reports.", 4.0),
    ],
}


class KeywordRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=5, score_threshold=-10.0):
        self.queries.append(query)
        out = []
        for key, docs in CORPUS.items():
            if key in query.lower():
                out.extend(docs)
        return out[:top_k]


def routed_llm(route=True, plan=None, reflect=None, answer="Answer [1]."):
    """LLM double dispatching on the prompt's ### Task header."""
    reflect = list(reflect or [])

    def fn(prompt, system):
        if "### Task: ROUTE" in prompt:
            return json.dumps({"retrieve": route, "reason": "test"})
        if "### Task: PLAN QUERIES" in prompt:
            return "```json\n" + json.dumps({"queries": plan or []}) + "\n```"
        if "### Task: REFLECT" in prompt:
            return json.dumps(reflect.pop(0)) if reflect else json.dumps({"sufficient": True})
        if "### Context" in prompt:
            return answer
        return "Hello! I can help with climate-law documents."

    return ScriptedLLM(fn=fn)


def test_parse_json_object_variants():
    assert parse_json_object('<think>x</think> {"retrieve": false}') == {"retrieve": False}
    assert parse_json_object("```json\n{'queries': ['a']}\n```") == {"queries": ["a"]}
    assert parse_json_object("no json here") is None
    assert parse_json_object("") is None


def test_citation_helpers():
    assert extract_citations("A [1]. B [2, 3]; C [1][4]") == [1, 2, 3, 4]
    cleaned, cited = sanitize_citations("A [1]. B [2, 9]. C [11] .", 3)
    assert cleaned == "A [1]. B [2]. C."
    assert cited == [1, 2]


def test_no_retrieval_for_small_talk():
    retriever = KeywordRetriever()
    resp = RetrievalAgent(retriever, routed_llm(route=False)).run("Is it sunny on Mars?")
    assert resp.used_retrieval is False
    assert retriever.queries == []
    assert resp.sources == [] and resp.citations == []
    assert resp.answer.startswith("Hello")


def test_retrieves_plans_and_cites_valid_chunks():
    retriever = KeywordRetriever()
    llm = routed_llm(plan=["STAN-METH-001 leakage", "double counting"], answer="Leakage is monitored [2] [7].")
    resp = RetrievalAgent(retriever, llm).run("How is leakage handled?")
    assert resp.used_retrieval
    assert retriever.queries == ["How is leakage handled?", "STAN-METH-001 leakage", "double counting"]
    # pool ranked by score: double counting chunk (6.0) first, leakage (5.0) second
    assert [s["chunk_id"] for s in resp.sources] == ["c2", "c1"]
    assert resp.answer == "Leakage is monitored [2]."
    assert [c["chunk_id"] for c in resp.citations] == ["c1"]
    assert resp.citations[0]["page"] == 2
    assert [s.action for s in resp.steps] == ["route", "plan", "retrieve", "reflect", "answer"]
    assert "[1] Source: doc2.pdf, page 3" in llm.prompts[-1]["prompt"]


def test_reflection_triggers_follow_up_queries():
    retriever = KeywordRetriever()
    llm = routed_llm(
        reflect=[{"sufficient": False, "missing": "reporting", "queries": ["corresponding adjustments"]}],
        answer="Adjustments avoid double counting [1]. They are reported in biennial transparency reports [2].",
    )
    resp = RetrievalAgent(retriever, llm, max_iterations=2).run("What about double counting?")
    assert retriever.queries == ["What about double counting?", "corresponding adjustments"]
    assert [s["chunk_id"] for s in resp.sources] == ["c2", "c3"]
    assert resp.sources[0]["score"] == 6.5  # best score kept on merge
    assert [c["chunk_id"] for c in resp.citations] == ["c2", "c3"]
    reflect_steps = [s for s in resp.steps if s.action == "reflect"]
    assert reflect_steps[0].detail["sufficient"] is False


def test_unparseable_router_defaults_to_retrieval_and_empty_pool():
    retriever = KeywordRetriever()
    llm = ScriptedLLM(fn=lambda p, s: "I am not JSON")
    resp = RetrievalAgent(retriever, llm).run("Unknown topic?")
    assert resp.used_retrieval is True
    assert retriever.queries == ["Unknown topic?"]
    assert resp.answer == INSUFFICIENT_CONTEXT
    assert resp.sources == []


@pytest.mark.parametrize("max_queries", [1, 2])
def test_query_budget(max_queries):
    retriever = KeywordRetriever()
    llm = routed_llm(plan=["leakage", "double counting", "x", "y"])
    RetrievalAgent(retriever, llm, max_queries=max_queries, max_iterations=1).run("Q")
    assert len(retriever.queries) == max_queries


def test_context_capped():
    retriever = KeywordRetriever()
    llm = routed_llm(plan=["leakage", "corresponding adjustments"])
    resp = RetrievalAgent(retriever, llm, max_context_chunks=2, max_iterations=1).run("double counting")
    assert len(resp.sources) == 2
    assert resp.to_dict()["sources"][0]["chunk_id"] == "c2"


def test_small_talk_fast_path_skips_llm_router():
    retriever, llm = KeywordRetriever(), routed_llm(route=True)
    resp = RetrievalAgent(retriever, llm).run("Hello, who are you?")
    assert resp.used_retrieval is False and retriever.queries == []
    assert not any("### Task: ROUTE" in p["prompt"] for p in llm.prompts)


def test_answer_prompt_has_citation_reminder_and_explicit_mode():
    llm = routed_llm(plan=["leakage"])
    resp = RetrievalAgent(KeywordRetriever(), llm).run("leakage?")
    assert "### Citation format" in llm.prompts[-1]["prompt"]
    assert resp.citation_mode == "explicit"


def test_inferred_citations_when_model_quotes_without_markers():
    chunks = [
        chunk(1, "Leakage shall be monitored under STAN-METH-001 for every removal activity.", 1.0),
        chunk(2, "Unrelated text about adaptation finance.", 0.5),
    ]
    answer = "The standard says leakage shall be monitored under STAN-METH-001 for every removal activity."
    assert infer_citations(answer, chunks) == [1]
    llm = routed_llm(plan=["leakage"], answer="Leakage shall be monitored under STAN-METH-001 per the standard.")
    resp = RetrievalAgent(KeywordRetriever(), llm).run("leakage?")
    assert resp.citation_mode == "inferred"
    assert [c["chunk_id"] for c in resp.citations] == ["c1"]


def test_verify_citations_repoints_unsupported_marker():
    chunks = [
        chunk(1, "Parties shall promote environmental integrity and transparency.", 1.0),
        chunk(2, "The host Party shall apply corresponding adjustments for first transfers of ITMOs.", 1.0),
    ]
    answer = "The host Party must apply corresponding adjustments for first transfers of ITMOs [1]. Integrity [1]."
    fixed, fixes = verify_citations(answer, chunks)
    assert fixes == [(1, 2)]
    assert (
        fixed == "The host Party must apply corresponding adjustments for first transfers of ITMOs [2]. Integrity [1]."
    )
    assert verify_citations("Integrity and transparency [1].", chunks) == ("Integrity and transparency [1].", [])


def test_agent_records_citation_remap():
    llm = routed_llm(plan=["leakage", "double counting"], answer="Leakage shall be monitored under STAN-METH-001 [1].")
    resp = RetrievalAgent(KeywordRetriever(), llm).run("leakage and double counting?")
    assert resp.answer.endswith("[2].")
    assert [c["chunk_id"] for c in resp.citations] == ["c1"]
    assert any(s.action == "verify_citations" for s in resp.steps)


def test_verify_citations_per_line_keeps_formatting_and_uses_idf():
    from lex6.agent import drop_contradictory_abstention
    from lex6.rag import INSUFFICIENT_CONTEXT

    chunks = [
        chunk(
            1, "Leakage - An increase in greenhouse gas emissions outside the boundary of a mitigation activity.", 1.0
        ),
        chunk(
            2,
            "Article 4 - Leakage. The Parties shall identify and deduct leakage. Where leakage cannot be "
            "quantified, a default deduction of ten percent of the mitigation outcomes shall apply.",
            1.0,
        ),
    ]
    answer = (
        "**Part 1: Analysis**\nThe agreement addresses leakage outside the boundary of a mitigation activity "
        "and, where leakage cannot be quantified, applies a default deduction of ten percent [1].\n\n**Part 2**"
    )
    fixed, fixes = verify_citations(answer, chunks)
    assert fixes == [(1, 2)]
    assert fixed == answer.replace("[1]", "[2]")

    assert drop_contradictory_abstention(f"Ten percent [2]. {INSUFFICIENT_CONTEXT}") == "Ten percent [2]."
    assert drop_contradictory_abstention(INSUFFICIENT_CONTEXT) == INSUFFICIENT_CONTEXT


REAL_LEAKAGE_ANSWER = """### Part 1: Analysis of the Agreement
The bilateral agreement does not contain specific provisions regarding the deduction.

### Part 2: Relevant Legal Standards
The relevant section from the glossary regarding leakage is as follows:
[4] Source: Article6_glossary.txt, page 1
Leakage - An increase in greenhouse gas emissions outside the boundary of a mitigation activity that is caused by that \
activity.

### Part 3: Legal Assessment
According to Article 4 of the bilateral agreement, the Parties shall identify, quantify, and deduct any increase in \
emissions occurring outside the boundary of a mitigation activity (leakage). Where leakage cannot be quantified, a \
default deduction of ten percent of the mitigation outcomes shall apply. Therefore, the default deduction of ten \
percent applies."""


def test_real_model_leakage_answer_is_recited_to_the_docx():
    """Regression: qwen2.5:3b copied a context header with a wrong number and quoted the DOCX without markers."""
    docs = [
        chunk(
            1,
            "Article 4 - Leakage\nThe Parties shall identify, quantify and deduct any increase in emissions "
            "occurring outside the boundary of a mitigation activity (leakage). Where leakage cannot be "
            "quantified, a default deduction of ten percent of the mitigation outcomes shall apply.",
            3.5,
        ),
        chunk(
            2,
            "Leakage - An increase in greenhouse gas emissions outside the boundary of a mitigation activity "
            "that is caused by that activity.",
            -2.6,
        ),
        chunk(3, "Article 1 - Purpose\nThe Parties establish a framework for cooperative approaches.", -5.8),
        chunk(4, "ITMO - Internationally transferred mitigation outcome.", -6.3),
    ]

    class Fixed:
        def retrieve(self, query, top_k=5, score_threshold=-10.0):
            return docs

    resp = RetrievalAgent(Fixed(), routed_llm(plan=[], answer=REAL_LEAKAGE_ANSWER)).run("leakage deduction?")
    assert "Source: Article6_glossary.txt" not in resp.answer
    assert "[4]" not in resp.answer
    assert "default deduction of ten percent of the mitigation outcomes shall apply [1]." in resp.answer
    assert "caused by that activity [2]." in resp.answer
    assert {c["chunk_id"] for c in resp.citations} == {"c1", "c2"}
    assert resp.citation_mode == "inferred"


def test_drop_uncited_abstention_that_contradicts_cited_answer():
    from lex6.agent import drop_contradictory_abstention

    q = "What is the registry fee for ITMO transfers?"
    answer = (
        "The registry fee for ITMO transfers is 37 euros per tonne [1]. However, the provided context does not "
        "contain information about the specific registry fee for ITMO transfers. It applies per tonne [1]."
    )
    assert drop_contradictory_abstention(answer, q) == (
        "The registry fee for ITMO transfers is 37 euros per tonne [1]. It applies per tonne [1]."
    )
    partial = "The fee is 37 euros [1]. The context does not specify when the fee is due."
    assert drop_contradictory_abstention(partial, q) == partial
    uncited = "The provided context does not contain information about the registry fee for ITMO transfers."
    assert drop_contradictory_abstention(uncited, q) == uncited


REVERSAL_CHUNKS = [
    {"id": "a", "content": "Bilateral agreement between Zambia and Sweden: authorization and transfer of ITMOs.",
     "metadata": {"source_file": "zambia_sweden.pdf"}, "score": 1.4},
    {"id": "b", "content": "The Parties agree on corresponding adjustments for each agreement.",
     "metadata": {"source_file": "agreement2.pdf"}, "score": 0.5},
    {"id": "c", "content": "4.6 Addressing reversals. On a reversal, units are cancelled from the buffer pool.",
     "metadata": {"source_file": "A6.4-STAN-METH-002_REMOVALS.pdf"}, "score": -3.6},
]  # fmt: skip
REVERSAL_Q = "How are reversals of carbon storage addressed in bilateral agreements?"


class _FixedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        return [dict(d) for d in self.chunks]


def _reversal_agent(answers):
    route_plan = ['{"retrieve": true, "reason": "legal"}', '{"queries": []}']
    llm = ScriptedLLM(responses=route_plan + answers)
    return RetrievalAgent(_FixedRetriever(REVERSAL_CHUNKS), llm, max_iterations=1), llm


def test_relevant_chunks_uses_distinctive_question_terms():
    from lex6.agent import is_abstention, relevant_chunks

    assert relevant_chunks(REVERSAL_Q, REVERSAL_CHUNKS) == ([1, 3], ["bilateral", "reversals"])
    assert relevant_chunks("What is the weather on Mars?", REVERSAL_CHUNKS) == ([], [])
    assert is_abstention(INSUFFICIENT_CONTEXT)
    assert is_abstention("The provided documents do not mention reversals.")
    assert not is_abstention("Reversals are remediated via the buffer pool [3].")


def test_abstention_with_relevant_chunks_is_retried():
    retry = "The bilateral agreements do not cover reversals; the A6.4 standard cancels units from the buffer pool [3]."
    agent, llm = _reversal_agent([INSUFFICIENT_CONTEXT, retry])
    resp = agent.run(REVERSAL_Q)
    assert "buffer pool [3]" in resp.answer and INSUFFICIENT_CONTEXT not in resp.answer
    assert [c["source"] for c in resp.citations] == ["A6.4-STAN-METH-002_REMOVALS.pdf"]
    step = next(s for s in resp.steps if s.action == "retry_answer")
    assert step.detail == {"relevant_chunks": [1, 3], "answered": True}
    assert "chunk(s) [1], [3] discuss bilateral, reversals" in llm.prompts[-1]["prompt"]
    assert "Partial answers" in llm.prompts[-1]["prompt"]


def test_abstention_kept_when_retry_also_abstains_or_nothing_relevant():
    agent, _ = _reversal_agent([INSUFFICIENT_CONTEXT, INSUFFICIENT_CONTEXT])
    resp = agent.run(REVERSAL_Q)
    assert resp.answer == INSUFFICIENT_CONTEXT and not resp.citations
    assert next(s for s in resp.steps if s.action == "retry_answer").detail["answered"] is False

    agent, llm = _reversal_agent([INSUFFICIENT_CONTEXT])
    resp = agent.run("What is the weather on Mars?")
    assert resp.answer == INSUFFICIENT_CONTEXT and len(llm.prompts) == 3
    assert not any(s.action == "retry_answer" for s in resp.steps)


def test_off_topic_cited_answer_becomes_abstention():
    padded = (
        "[2] The Parties agree on corresponding adjustments.\n"
        "The agreements do not address the population of Mars colonies."
    )
    agent, _ = _reversal_agent([padded])
    resp = agent.run("What is the population of Mars colonies in 2050?")
    assert resp.answer == INSUFFICIENT_CONTEXT and not resp.citations
    assert next(s for s in resp.steps if s.action == "abstain").detail["dropped_citations"] == [2]


def test_partial_answer_with_on_topic_context_is_kept():
    partial = (
        "The bilateral agreements do not themselves contain reversal rules. Under the removals standard, "
        "reversed units are cancelled from the buffer pool [3]."
    )
    agent, _ = _reversal_agent([partial])
    resp = agent.run(REVERSAL_Q)
    assert [c["index"] for c in resp.citations] == [3]
    assert "buffer pool" in resp.answer
    assert not any(s.action == "abstain" for s in resp.steps)
