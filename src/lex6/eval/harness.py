"""Run a fixed question set through two systems and print a comparison table.

Systems
  * ``baseline``: notebook ``rag_advanced`` with the naive dense ``RAGRetriever`` - one retrieval with the raw
    question, one generation, no routing/reflection/citation checks.
  * ``agent``: ``RetrievalAgent`` (routing, query planning, HyDE + BM25 + rerank, reflection, cited answers).

Metrics (deterministic, no LLM judge), each in [0, 1]; a question only gets the metrics that apply to its type:
  * ``source_recall``      share of expected source groups present among the chunks given to the LLM
  * ``keyword_coverage``   share of expected keyword groups present in the answer
  * ``citation_precision`` share of cited chunks that come from an expected source (0 if nothing is cited)
  * ``abstention``         answerable -> did not abstain; unanswerable -> abstained
  * ``routing``            retrieval used iff the question needs it
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from tabulate import tabulate

from lex6.agent import RetrievalAgent, extract_citations
from lex6.llm import LLM
from lex6.rag import rag_advanced
from lex6.retrieval.base import Retriever

DEFAULT_QUESTIONS = Path(__file__).with_name("questions.json")
METRICS = ["source_recall", "keyword_coverage", "citation_precision", "abstention", "routing"]
ABSTAIN_MARKERS = (
    "does not contain sufficient information",
    "no relevant context found",
    "not contain enough information",
    "cannot be answered from",
)


@dataclass
class EvalQuestion:
    id: str
    question: str
    type: str = "answerable"  # answerable | unanswerable | no_retrieval
    expected_sources: list[list[str]] = field(default_factory=list)
    expected_keywords: list[list[str]] = field(default_factory=list)

    @property
    def needs_retrieval(self) -> bool:
        return self.type != "no_retrieval"


@dataclass
class SystemOutput:
    answer: str
    sources: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    used_retrieval: bool
    latency_s: float = 0.0


def load_questions(path: str | Path | None = None) -> list[EvalQuestion]:
    with open(path or DEFAULT_QUESTIONS, encoding="utf-8") as f:
        return [EvalQuestion(**q) for q in json.load(f)]


class NaiveBaseline:
    name = "baseline"

    def __init__(self, retriever: Retriever, llm: LLM, top_k: int = 5, min_score: float = 0.0):
        self.retriever, self.llm, self.top_k, self.min_score = retriever, llm, top_k, min_score

    def __call__(self, question: str) -> SystemOutput:
        out = rag_advanced(question, self.retriever, self.llm, top_k=self.top_k, min_score=self.min_score)
        sources = out["sources"]
        cited = [n for n in extract_citations(out["answer"]) if 1 <= n <= len(sources)]
        return SystemOutput(out["answer"], sources, [sources[n - 1] for n in cited], used_retrieval=True)


class AgentSystem:
    name = "agent"

    def __init__(self, agent: RetrievalAgent):
        self.agent = agent

    def __call__(self, question: str) -> SystemOutput:
        r = self.agent.run(question)
        return SystemOutput(r.answer, r.sources, r.citations, r.used_retrieval)


def _contains(text: str, options: list[str]) -> bool:
    low = text.lower()
    return any(o.lower() in low for o in options)


def is_abstention(answer: str) -> bool:
    return _contains(answer, list(ABSTAIN_MARKERS))


def score_output(q: EvalQuestion, out: SystemOutput) -> dict[str, float]:
    scores: dict[str, float] = {"routing": float(out.used_retrieval == q.needs_retrieval)}
    if q.type == "no_retrieval":
        return scores
    abstained = is_abstention(out.answer)
    if q.type == "unanswerable":
        scores["abstention"] = float(abstained)
        return scores
    scores["abstention"] = float(not abstained)
    retrieved = " ".join(s["source"] for s in out.sources)
    if q.expected_sources:
        scores["source_recall"] = mean(float(_contains(retrieved, g)) for g in q.expected_sources)
        expected = [name for g in q.expected_sources for name in g]
        scores["citation_precision"] = (
            mean(float(_contains(c["source"], expected)) for c in out.citations) if out.citations else 0.0
        )
    if q.expected_keywords:
        scores["keyword_coverage"] = mean(float(_contains(out.answer, g)) for g in q.expected_keywords)
    return scores


def overall(scores: dict[str, float]) -> float:
    return mean(scores.values()) if scores else 0.0


def run_eval(
    questions: list[EvalQuestion],
    systems: dict[str, Callable[[str], SystemOutput]],
    on_result: Callable[[str, str, dict], None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for q in questions:
        row: dict[str, Any] = {"id": q.id, "type": q.type, "question": q.question, "results": {}}
        for name, system in systems.items():
            start = time.perf_counter()
            try:
                out = system(q.question)
                error = None
            except Exception as e:  # one failing question must not abort the whole run
                out, error = SystemOutput(f"ERROR: {e}", [], [], used_retrieval=False), str(e)
            out.latency_s = time.perf_counter() - start
            scores = score_output(q, out)
            result = {"scores": scores, "overall": overall(scores), "output": asdict(out), "error": error}
            row["results"][name] = result
            if on_result:
                on_result(q.id, name, result)
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], systems: list[str]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for metric in [*METRICS, "overall", "latency_s"]:
        summary[metric] = {}
        for s in systems:
            if metric == "overall":
                vals = [r["results"][s]["overall"] for r in rows]
            elif metric == "latency_s":
                vals = [r["results"][s]["output"]["latency_s"] for r in rows]
            else:
                vals = [r["results"][s]["scores"][metric] for r in rows if metric in r["results"][s]["scores"]]
            summary[metric][s] = mean(vals) if vals else None
    return summary


def format_report(rows: list[dict[str, Any]], systems: list[str]) -> str:
    def fmt(v: float | None, digits: int = 2) -> str:
        return "-" if v is None else f"{v:.{digits}f}"

    per_q = [
        [r["id"], r["type"], *(fmt(r["results"][s]["overall"]) for s in systems)]
        + (
            [fmt(r["results"][systems[1]]["overall"] - r["results"][systems[0]]["overall"], 2)]
            if len(systems) == 2
            else []
        )
        for r in rows
    ]
    delta_hdr = [f"Δ ({systems[1]}-{systems[0]})"] if len(systems) == 2 else []
    summary = summarize(rows, systems)
    sum_rows = []
    for metric, vals in summary.items():
        row = [metric, *(fmt(vals[s], 1 if metric == "latency_s" else 3) for s in systems)]
        if len(systems) == 2:
            a, b = vals[systems[0]], vals[systems[1]]
            row.append("-" if a is None or b is None else f"{b - a:+.{1 if metric == 'latency_s' else 3}f}")
        sum_rows.append(row)
    return (
        "Per-question overall score\n"
        + tabulate(per_q, headers=["id", "type", *systems, *delta_hdr], tablefmt="github", disable_numparse=True)
        + "\n\nComparison (mean over applicable questions)\n"
        + tabulate(sum_rows, headers=["metric", *systems, *delta_hdr], tablefmt="github", disable_numparse=True)
    )
