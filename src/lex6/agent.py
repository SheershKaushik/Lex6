"""Retrieval agent: decides whether to retrieve, plans queries, reflects, answers with chunk citations.

Loop::

    route(question)                -> answer directly (no retrieval) | retrieve
    plan_queries(question)         -> 1..max_queries search queries (always includes the question)
    retrieve each query            -> merged chunk pool (AdvancedRAGRetriever: HyDE + BM25 + rerank)
    reflect(question, chunks)      -> sufficient? else follow-up queries (up to max_iterations rounds)
    answer(question, chunks)       -> legal assessment with [n] citations, validated against the pool
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from lex6.llm import LLM
from lex6.rag import INSUFFICIENT_CONTEXT, LEGAL_ASSESSMENT_PROMPT, build_sources, confidence, format_context
from lex6.retrieval.base import Retriever

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """### Task: ROUTE
You are the router of a question-answering assistant over a private library of climate-law documents
(treaties, bilateral Article 6 agreements, standards, NDCs). Decide whether answering the user message
requires searching that library.

Answer "retrieve": false ONLY for greetings, small talk, questions about the assistant itself, or
pure arithmetic/wording tasks that need no document facts. Any question about agreements, articles,
obligations, definitions, countries, standards or anything that could be in the documents needs retrieval.

Respond with JSON only: {{"retrieve": true|false, "reason": "<short reason>"}}

User message: {question}
"""

PLANNER_PROMPT = """### Task: PLAN QUERIES
Rewrite the user question into at most {max_queries} short, self-contained search queries for a hybrid
keyword + semantic search over climate-law documents. Split multi-part questions into separate queries,
expand abbreviations (e.g. ITMO = internationally transferred mitigation outcome) and keep article
numbers / standard codes verbatim.

Respond with JSON only: {{"queries": ["...", "..."]}}

Question: {question}
"""

REFLECT_PROMPT = """### Task: REFLECT
Decide whether the numbered context chunks below contain enough information to fully answer the question.
If not, propose up to {max_queries} NEW search queries for the missing information (different from: {tried}).

Respond with JSON only: {{"sufficient": true|false, "missing": "<what is missing>", "queries": ["..."]}}

### Context
{context}

### Question
{question}
"""

DIRECT_PROMPT = """You are Lex6, an assistant for climate-law document analysis. The following message does not
need the document library. Reply briefly and helpfully. If it actually asks about document contents, say
that you need to search the documents.

Message: {question}
"""

CITATION_REMINDER = (
    "### Citation format\nAfter every claim put the number of the context chunk it comes from in square "
    "brackets, e.g. [1] or [2, 3]. Cite chunk numbers, not file names, and do not copy the chunk headers. "
    "Read every chunk before saying the context lacks something; never claim information is missing and then "
    "use it.\n\n### Answer"
)

_SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks?( you)?|thank you|bye|goodbye|"
    r"who are you|what are you|what can you do|how are you)\b[\s\w,!?.']{0,40}$",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object in an LLM response."""
    if not text:
        return None
    match = _JSON_RE.search(text.replace("```json", "").replace("```", ""))
    if not match:
        return None
    candidate = match.group(0)
    for attempt in (candidate, candidate.replace("'", '"')):
        try:
            value = json.loads(attempt)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "yes", "no"}:
        return value.strip().lower() in {"true", "yes"}
    return default


def extract_citations(answer: str) -> list[int]:
    numbers: list[int] = []
    for group in _CITATION_RE.findall(answer):
        for n in re.split(r"\s*[,;]\s*", group):
            if int(n) not in numbers:
                numbers.append(int(n))
    return numbers


def sanitize_citations(answer: str, n_chunks: int) -> tuple[str, list[int]]:
    """Drop citation markers that point outside ``1..n_chunks``; return the cleaned answer + valid ids."""

    def repl(match: re.Match) -> str:
        kept = [n for n in re.split(r"\s*[,;]\s*", match.group(1)) if 1 <= int(n) <= n_chunks]
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _CITATION_RE.sub(repl, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return cleaned, [n for n in extract_citations(cleaned) if 1 <= n <= n_chunks]


def infer_citations(answer: str, chunks: list[dict[str, Any]], shingle: int = 8) -> list[int]:
    """Fallback when the model cites no ``[n]``: chunks quoted verbatim (``shingle`` words) in the answer."""
    answer_tokens = re.findall(r"\w+", answer.lower())
    answer_grams = {tuple(answer_tokens[i : i + shingle]) for i in range(len(answer_tokens) - shingle + 1)}
    cited = []
    for i, doc in enumerate(chunks, start=1):
        tokens = re.findall(r"\w+", doc["content"].lower())
        if any(tuple(tokens[j : j + shingle]) in answer_grams for j in range(len(tokens) - shingle + 1)):
            cited.append(i)
    return cited


_SOURCE_ECHO_RE = re.compile(r"^[ \t]*\[\d+(?:\s*,\s*\d+)*\][ \t]*Source:.*(?:\n|$)", re.MULTILINE)


def strip_source_echoes(answer: str) -> str:
    """Remove copied context headers ("[4] Source: file, page 1"); their numbers are often wrong."""
    return _SOURCE_ECHO_RE.sub("", answer)


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.lower())
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def add_missing_citations(answer: str, chunks: list[dict[str, Any]], shingle: int = 8) -> tuple[str, list[int]]:
    """Append ``[m]`` to uncited sentences/lines that quote chunk ``m`` verbatim (``shingle`` words)."""
    chunk_grams = [_shingles(d["content"], shingle) for d in chunks]
    added: list[int] = []
    parts = re.split(r"((?<=[.!?])[ \t]+|\n+)", answer)
    for k in range(0, len(parts), 2):
        sentence = parts[k]
        if not sentence.strip() or _CITATION_RE.search(sentence):
            continue
        grams = _shingles(sentence, shingle)
        nums = [i for i, cg in enumerate(chunk_grams, start=1) if grams & cg]
        if not nums:
            continue
        marker = f" [{', '.join(map(str, nums))}]"
        stripped = sentence.rstrip()
        trail = sentence[len(stripped) :]
        if stripped[-1] in ".!?;:":
            parts[k] = stripped[:-1] + marker + stripped[-1] + trail
        else:
            parts[k] = stripped + marker + trail
        added.extend(n for n in nums if n not in added)
    return ("".join(parts) if added else answer), added


def _content_words(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 3}


def verify_citations(
    answer: str, chunks: list[dict[str, Any]], min_support: float = 0.5, margin: float = 0.25
) -> tuple[str, list[tuple[int, int]]]:
    """Re-point a sentence's ``[n]`` to the chunk that actually supports it.

    Small local models often attach the wrong number. Support = IDF-weighted share of the sentence's content
    words found in a chunk (words shared by many chunks count less). A marker moves from chunk ``n`` to chunk
    ``m`` when ``m`` has at least ``min_support`` and beats ``n`` by ``margin``. Sentences and lines are checked
    separately; formatting is preserved.
    """
    chunk_words = [_content_words(d["content"]) for d in chunks]
    n_chunks = len(chunks)
    df: dict[str, int] = {}
    for cw in chunk_words:
        for w in cw:
            df[w] = df.get(w, 0) + 1

    def idf(w: str) -> float:
        return math.log(1 + n_chunks / df[w])

    fixes: list[tuple[int, int]] = []
    parts = re.split(r"((?<=[.!?])[ \t]+|\n+)", answer)
    for k in range(0, len(parts), 2):
        sentence = parts[k]
        # Paraphrase words found in no chunk can't discriminate between chunks.
        words = {w for w in _content_words(_CITATION_RE.sub("", sentence)) if w in df}
        if not words or not _CITATION_RE.search(sentence):
            continue
        total = sum(idf(w) for w in words)
        support = [sum(idf(w) for w in words & cw) / total for cw in chunk_words]
        best = max(range(n_chunks), key=lambda i: support[i]) + 1

        def repl(match: re.Match, support=support, best=best) -> str:
            nums = []
            for n in (int(x) for x in re.split(r"\s*[,;]\s*", match.group(1))):
                if support[best - 1] >= min_support and support[best - 1] - support[n - 1] >= margin:
                    fixes.append((n, best))
                    n = best
                if n not in nums:
                    nums.append(n)
            return f"[{', '.join(map(str, nums))}]"

        parts[k] = _CITATION_RE.sub(repl, sentence)
    return ("".join(parts) if fixes else answer), fixes


def drop_contradictory_abstention(answer: str) -> str:
    """Remove the 'insufficient information' boilerplate when the model answered (with citations) anyway."""
    if INSUFFICIENT_CONTEXT not in answer or not _CITATION_RE.search(answer):
        return answer
    stripped = answer.replace(INSUFFICIENT_CONTEXT, "").strip()
    return stripped or answer


@dataclass
class AgentStep:
    action: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    question: str
    answer: str
    used_retrieval: bool
    queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)  # every chunk given to the LLM
    citations: list[dict[str, Any]] = field(default_factory=list)  # chunks actually cited in the answer
    confidence: float = 0.0
    citation_mode: str = "none"  # explicit ([n] markers) | inferred (verbatim quotes) | none
    steps: list[AgentStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievalAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        top_k: int = 5,
        max_context_chunks: int = 8,
        max_queries: int = 3,
        max_iterations: int = 2,
        score_threshold: float = -10.0,
        control_max_tokens: int = 512,
        answer_max_tokens: int | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.max_queries = max_queries
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.control_max_tokens = control_max_tokens
        self.answer_max_tokens = answer_max_tokens

    # --- decisions ---------------------------------------------------------------------
    def route(self, question: str) -> tuple[bool, str]:
        if _SMALL_TALK_RE.match(question):
            return False, "small talk"
        raw = self.llm.generate(ROUTER_PROMPT.format(question=question), max_tokens=self.control_max_tokens)
        data = parse_json_object(raw)
        if data is None:
            return True, "router output unparseable; defaulting to retrieval"
        return _as_bool(data.get("retrieve"), True), str(data.get("reason", ""))

    def plan_queries(self, question: str) -> list[str]:
        raw = self.llm.generate(
            PLANNER_PROMPT.format(question=question, max_queries=self.max_queries),
            max_tokens=self.control_max_tokens,
        )
        data = parse_json_object(raw) or {}
        planned = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        return _dedupe([question, *planned])[: self.max_queries]

    def reflect(self, question: str, chunks: list[dict[str, Any]], tried: list[str]) -> tuple[bool, list[str], str]:
        raw = self.llm.generate(
            REFLECT_PROMPT.format(
                question=question,
                context=format_context(chunks),
                max_queries=self.max_queries,
                tried=json.dumps(tried),
            ),
            max_tokens=self.control_max_tokens,
        )
        data = parse_json_object(raw)
        if data is None:
            return True, [], "reflection output unparseable; answering with current context"
        new = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip() and q not in tried]
        sufficient = _as_bool(data.get("sufficient"), True) or not new
        return sufficient, new[: self.max_queries], str(data.get("missing", ""))

    # --- retrieval ---------------------------------------------------------------------
    def _search(self, queries: list[str], pool: dict[str, dict[str, Any]]) -> None:
        for q in queries:
            for doc in self.retriever.retrieve(q, top_k=self.top_k, score_threshold=self.score_threshold):
                current = pool.get(doc["id"])
                if current is None or _rank_key(doc) > _rank_key(current):
                    pool[doc["id"]] = {**doc, "query": q}

    def _context(self, pool: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(pool.values(), key=_rank_key, reverse=True)
        country = sum(1 for d in ranked if d.get("country_match"))
        return ranked[: max(self.max_context_chunks, country)]

    # --- main entry point --------------------------------------------------------------
    def run(self, question: str) -> AgentResponse:
        question = question.strip()
        steps: list[AgentStep] = []
        needs_retrieval, reason = self.route(question)
        steps.append(AgentStep("route", {"retrieve": needs_retrieval, "reason": reason}))
        if not needs_retrieval:
            answer = self.llm.generate(DIRECT_PROMPT.format(question=question), max_tokens=self.control_max_tokens)
            return AgentResponse(question, answer.strip(), used_retrieval=False, steps=steps)

        queries = self.plan_queries(question)
        steps.append(AgentStep("plan", {"queries": queries}))
        pool: dict[str, dict[str, Any]] = {}
        tried: list[str] = []
        pending = queries
        for iteration in range(1, self.max_iterations + 1):
            self._search(pending, pool)
            tried.extend(pending)
            steps.append(AgentStep("retrieve", {"iteration": iteration, "queries": pending, "pool_size": len(pool)}))
            if iteration == self.max_iterations or not pool:
                break
            sufficient, pending, missing = self.reflect(question, self._context(pool), tried)
            steps.append(AgentStep("reflect", {"sufficient": sufficient, "missing": missing, "queries": pending}))
            if sufficient:
                break

        context = self._context(pool)
        if not context:
            steps.append(AgentStep("answer", {"chunks": 0}))
            return AgentResponse(question, INSUFFICIENT_CONTEXT, True, tried, steps=steps)

        prompt = LEGAL_ASSESSMENT_PROMPT.format(
            context=format_context(context), question=question, insufficient=INSUFFICIENT_CONTEXT
        ).replace("### Answer", CITATION_REMINDER)
        raw_answer = self.llm.generate(prompt, max_tokens=self.answer_max_tokens)
        answer, cited = sanitize_citations(strip_source_echoes(raw_answer), len(context))
        mode = "explicit" if cited else "none"
        if cited:
            answer, fixes = verify_citations(answer, context)
            if fixes:
                steps.append(AgentStep("verify_citations", {"remapped": fixes}))
        answer, added = add_missing_citations(answer, context)
        if added:
            steps.append(AgentStep("add_citations", {"quoted_chunks": added}))
            mode = mode if cited else "inferred"
        cited = extract_citations(answer)
        if cited:
            answer = drop_contradictory_abstention(answer)
        sources = build_sources(context)
        steps.append(AgentStep("answer", {"chunks": len(context), "cited": cited, "citation_mode": mode}))
        return AgentResponse(
            question=question,
            answer=answer.strip(),
            used_retrieval=True,
            queries=tried,
            sources=sources,
            citations=[sources[n - 1] for n in cited],
            confidence=confidence(context),
            citation_mode=mode,
            steps=steps,
        )


def _rank_key(doc: dict[str, Any]) -> tuple[bool, float]:
    """Country-matched chunks rank above everything else (cross-encoder logits can exceed their 1.0 score)."""
    return bool(doc.get("country_match")), doc.get("score") or 0.0


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
