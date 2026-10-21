"""RAG pipelines from the notebook (``rag_simple``, ``rag_advanced``, ``AdvancedRAGPipeline``).

Context chunks are numbered ``[1]..[n]`` so answers can cite the exact source chunk.
"""

from __future__ import annotations

import logging
from typing import Any

from lex6.llm import LLM
from lex6.retrieval.base import Retriever, page_label, source_name

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "No relevant context found to answer the question."
INSUFFICIENT_CONTEXT = "The provided context does not contain sufficient information to answer this question."

LEGAL_ASSESSMENT_PROMPT = """### Instruction
You are an expert legal assistant specializing in Climate Law. Your task is to analyze the provided context \
and answer the user's question following a strict legal assessment structure.

### Guidelines
1. **Source Truth:** Answer using ONLY the information provided in the "Context" section below. \
Do not use outside knowledge.
2. **Tone:** Maintain a professional, objective legal tone. Use precise terminology suitable for a legal professional.
3. **Citations:** Every claim must cite the numbered context chunk(s) it relies on using square brackets, \
e.g. "Parties shall pursue domestic mitigation measures [2]." Only cite numbers that appear in the Context.
4. **Partial answers:** If the context only partly answers the question (e.g. the rule is in a standard or \
guidance rather than in an agreement), answer with what it does say, citing it, and state briefly which part \
is not covered.
5. **Fallback:** Only if no chunk in the context is relevant to the question, state exactly: "{insufficient}" \
Do not attempt to make up an answer.

### Response Structure
You must provide your answer in three distinct parts. Do not skip any steps.

**Part 1: Analysis of the Agreement**
Identify and quote the specific sections from the provided agreement or document that are relevant to the question.

**Part 2: Relevant Legal Standards**
Identify and quote the relevant sections from the provided standards or legal materials \
(e.g., ISO standards, regulations) that apply to the situation.

**Part 3: Legal Assessment**
Based on the evidence gathered in Part 1 and Part 2, provide a clear and professional legal conclusion. \
Synthesize the information to answer the user's question directly.

### Context
{context}

### Question
{question}

### Answer
"""

SUMMARY_PROMPT = "Summarize the following answer in 2 sentences:\n{answer}"


def format_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for i, doc in enumerate(results, start=1):
        meta = doc["metadata"]
        header = f"[{i}] Source: {source_name(meta)}"
        page = page_label(meta)
        if page is not None:
            header += f", page {page}"
        if meta.get("translated"):
            header += f" (translated from {meta.get('original_language')})"
        blocks.append(f"{header}\n{doc['content']}")
    return "\n\n".join(blocks)


def build_sources(results: list[dict[str, Any]], preview_length: int = 300) -> list[dict[str, Any]]:
    sources = []
    for i, doc in enumerate(results, start=1):
        meta = doc["metadata"]
        content = doc["content"]
        sources.append(
            {
                "index": i,
                "chunk_id": doc["id"],
                "document_id": meta.get("document_id"),
                "source": source_name(meta),
                "page": page_label(meta),
                "score": doc.get("score"),
                "similarity_score": doc.get("similarity_score"),
                "retrieval_source": doc.get("retrieval_source"),
                "original_language": meta.get("original_language"),
                "translated": bool(meta.get("translated")),
                "preview": content[:preview_length] + ("..." if len(content) > preview_length else ""),
            }
        )
    return sources


def confidence(results: list[dict[str, Any]]) -> float:
    sims = [d["similarity_score"] for d in results if d.get("similarity_score") is not None]
    return float(max(sims)) if sims else 0.0


def generate_answer(llm: LLM, question: str, results: list[dict[str, Any]], max_tokens: int | None = None) -> str:
    prompt = LEGAL_ASSESSMENT_PROMPT.format(
        context=format_context(results), question=question, insufficient=INSUFFICIENT_CONTEXT
    )
    return llm.generate(prompt, max_tokens=max_tokens)


def rag_simple(query: str, retriever: Retriever, llm: LLM, top_k: int = 3) -> str:
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return NO_CONTEXT_ANSWER
    return generate_answer(llm, query, results)


def rag_advanced(
    query: str,
    retriever: Retriever,
    llm: LLM,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> dict[str, Any]:
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        out: dict[str, Any] = {"answer": NO_CONTEXT_ANSWER, "sources": [], "confidence": 0.0}
        if return_context:
            out["context"] = ""
        return out
    out = {
        "answer": generate_answer(llm, query, results),
        "sources": build_sources(results),
        "confidence": confidence(results),
    }
    if return_context:
        out["context"] = format_context(results)
    return out


class AdvancedRAGPipeline:
    """Retrieve -> answer with numbered citations -> optional summary, keeping a query history."""

    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm
        self.history: list[dict[str, Any]] = []

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, summarize: bool = False) -> dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if results:
            answer = generate_answer(self.llm, question, results)
            sources = build_sources(results, preview_length=120)
        else:
            answer, sources = NO_CONTEXT_ANSWER, []
        citations = [
            f"[{s['index']}] {s['source']}" + (f" (page {s['page']})" if s["page"] is not None else "") for s in sources
        ]
        summary = None
        if summarize and results:
            summary = self.llm.generate(SUMMARY_PROMPT.format(answer=answer), max_tokens=256)
        entry = {"question": question, "answer": answer, "sources": sources, "summary": summary}
        self.history.append(entry)
        return {**entry, "citations": citations, "history": self.history}
