"""HTTP API for ``three.html``.

Implements the routes the Upload and Ask views call. Compare, Reports export and the Knowledge Graph are
deliberately not implemented yet (their buttons get a 404).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lex6.agent import AgentResponse, RetrievalAgent
from lex6.config import Settings, get_settings
from lex6.ingestion.pipeline import IngestionPipeline
from lex6.llm import LLMError, check_model
from lex6.retrieval.base import Retriever, page_label
from lex6.vectorstore import VectorStore

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")


class AssessmentRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    max_tokens: int | None = Field(default=None, ge=64, le=32768)
    preview_length: int | None = None


class DocumentFilterRetriever:
    """Restricts any retriever to one document by over-fetching and filtering on ``metadata.document_id``."""

    def __init__(self, base: Retriever, document_id: str, overfetch: int = 4):
        self.base, self.document_id, self.overfetch = base, document_id, overfetch

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        results = self.base.retrieve(query, top_k=top_k * self.overfetch, score_threshold=score_threshold)
        return [r for r in results if r["metadata"].get("document_id") == self.document_id][:top_k]


AgentFactory = Callable[[Retriever, int | None], RetrievalAgent]


class Lex6Service:
    """Owns the vector store, ingestion pipeline and agent; serialises access to the (non thread-safe) index."""

    def __init__(
        self,
        vector_store: VectorStore,
        pipeline: IngestionPipeline,
        retriever: Retriever,
        agent_factory: AgentFactory,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store, self.pipeline, self.retriever = vector_store, pipeline, retriever
        self.agent_factory = agent_factory
        self.assessments: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Lex6Service:
        from lex6.services import build_ingestion_pipeline, build_llm, build_retriever, build_vector_store

        s = settings or get_settings()
        vs = build_vector_store(s)
        llm = build_llm(s)

        def agent_factory(retriever: Retriever, max_tokens: int | None) -> RetrievalAgent:
            return RetrievalAgent(retriever, llm, answer_max_tokens=max_tokens)

        return cls(vs, build_ingestion_pipeline(s, vs), build_retriever("advanced", s, vs), agent_factory, s)

    # --- documents -------------------------------------------------------------------
    def ingest_uploads(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        batch_dir = self.settings.uploads_dir / uuid.uuid4().hex[:12]
        batch_dir.mkdir(parents=True, exist_ok=True)
        paths, names = [], {}
        for raw_name, data in files:
            name = Path(raw_name or "upload").name or "upload"
            path = batch_dir / name
            path.write_bytes(data)
            paths.append(path)
            names[path] = name
        with self.lock:
            reports = self.pipeline.ingest(paths, display_names=names)
        return {
            "document_ids": [r.document_id for r in reports if r.status == "ingested"],
            "reports": [r.to_dict() for r in reports],
        }

    def chunks_for_page(self, document_id: str, page_number: int) -> list[dict[str, Any]]:
        return [
            self._clause(c)
            for c in self.vector_store.all_chunks()
            if c["metadata"].get("document_id") == document_id and page_label(c["metadata"]) == page_number
        ]

    def chunk(self, chunk_id: str) -> dict[str, Any] | None:
        return next((self._clause(c) for c in self.vector_store.all_chunks() if c["original_id"] == chunk_id), None)

    @staticmethod
    def _clause(c: dict[str, Any]) -> dict[str, Any]:
        md = c["metadata"]
        return {
            "id": c["original_id"],
            "document_id": md.get("document_id"),
            "page_number": page_label(md),
            "section_label": f"chunk {md.get('chunk_index', '')}".strip(),
            "text": c["text"],
            "topics": [],
        }

    # --- ask -------------------------------------------------------------------------
    def assess(self, req: AssessmentRequest) -> dict[str, Any]:
        retriever: Retriever = self.retriever
        if req.document_id:
            if not self.vector_store.get_document(req.document_id):
                raise KeyError(req.document_id)
            retriever = DocumentFilterRetriever(retriever, req.document_id)
        with self.lock:
            response = self.agent_factory(retriever, req.max_tokens).run(req.query)
        result = to_assessment(response, req, preview_length=req.preview_length or 300)
        self.assessments.append(
            {"id": result["assessment_id"], "query": req.query, "created_at": result["created_at"], **result}
        )
        return result


def _source_line(s: dict[str, Any], preview_length: int) -> str:
    page = f", p. {s['page']}" if s.get("page") else ""
    preview = s.get("preview", "")[:preview_length]
    return f"[{s['index']}] {s['source']}{page} (score {s['score']:.2f}): {preview}"


def to_assessment(r: AgentResponse, req: AssessmentRequest, preview_length: int = 300) -> dict[str, Any]:
    """Map an ``AgentResponse`` onto the shape three.html renders (``short_answer`` + ``detailed_explanation``)."""
    details: list[str] = []
    if r.used_retrieval:
        cited = {c["index"] for c in r.citations}
        if r.citations:
            details.append("Cited sources:\n" + "\n".join(_source_line(c, preview_length) for c in r.citations))
        other = [s for s in r.sources if s["index"] not in cited]
        if other:
            details.append("Other retrieved context:\n" + "\n".join(_source_line(s, 120) for s in other))
        details.append(f"Search queries: {'; '.join(r.queries)}")
        details.append(f"Confidence: {r.confidence:.2f} | citations: {r.citation_mode}")
    else:
        details.append("Answered directly - the agent decided no document retrieval was needed.")
    details.append("Agent steps: " + " -> ".join(s.action for s in r.steps))
    return {
        "assessment_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": r.question,
        "document_id": req.document_id,
        "short_answer": r.answer,
        "answer": r.answer,
        "detailed_explanation": details,
        "evidence": r.citations or r.sources,
        "citations": r.citations,
        "sources": r.sources,
        "used_retrieval": r.used_retrieval,
        "queries": r.queries,
        "confidence": r.confidence,
        "citation_mode": r.citation_mode,
        "steps": [{"action": s.action, "detail": s.detail} for s in r.steps],
    }


def create_app(service: Lex6Service | None = None, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Lex6 agentic RAG")
    state: dict[str, Lex6Service] = {}

    def svc() -> Lex6Service:
        if "service" not in state:
            state["service"] = service or Lex6Service.from_settings(settings)
        return state["service"]

    if service is not None:
        state["service"] = service

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        s = svc()
        llm_problem = check_model(s.settings.llm_model, s.settings.ollama_base_url)
        return {
            "status": "ok",
            "llm_model": s.settings.llm_model,
            "llm": {"model": s.settings.llm_model, "ok": llm_problem is None, "error": llm_problem},
            "documents": len(s.vector_store.list_documents()),
            "chunks": s.vector_store.ntotal,
        }

    @app.get("/api/documents")
    def list_documents() -> list[dict[str, Any]]:
        return svc().vector_store.list_documents()

    @app.post("/api/documents")
    def upload(files: Annotated[list[UploadFile], File()]) -> dict[str, Any]:
        if not files:
            raise HTTPException(400, "no files uploaded")
        return svc().ingest_uploads([(f.filename or "upload", f.file.read()) for f in files])

    @app.get("/api/documents/{document_id}/pages")
    def pages(document_id: str) -> list[dict[str, Any]]:
        s = svc()
        if not s.vector_store.get_document(document_id):
            raise HTTPException(404, "unknown document")
        return s.vector_store.get_pages(document_id)

    @app.get("/api/documents/{document_id}/pages/{page_number}/clauses")
    def page_chunks(document_id: str, page_number: int) -> list[dict[str, Any]]:
        return svc().chunks_for_page(document_id, page_number)

    @app.get("/api/clauses/{chunk_id}")
    def clause(chunk_id: str) -> dict[str, Any]:
        c = svc().chunk(chunk_id)
        if c is None:
            raise HTTPException(404, "unknown chunk")
        return c

    @app.post("/api/assessments")
    def assess(req: AssessmentRequest) -> dict[str, Any]:
        try:
            return svc().assess(req)
        except KeyError as e:
            raise HTTPException(404, f"unknown document {e}") from e
        except LLMError as e:
            logger.warning("assessment failed: %s", e)
            raise HTTPException(503, str(e)) from e

    @app.get("/api/assessments")
    def list_assessments() -> list[dict[str, Any]]:
        return [{"id": a["id"], "query": a["query"], "created_at": a["created_at"]} for a in svc().assessments]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
