"""HTTP API for ``three.html``.

Implements the routes the Upload and Ask views call. Compare, Reports export and the Knowledge Graph are
deliberately not implemented yet (their buttons get a 404).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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
ACCESS_COOKIE = "lex6_access"
PUBLIC_PATHS = {"/login", "/api/login", "/static/du.png"}


class LoginRequest(BaseModel):
    passcode: str


def passcode_matches(passcode: str, expected_sha256: str) -> bool:
    digest = hashlib.sha256(passcode.encode()).hexdigest()
    return hmac.compare_digest(digest, expected_sha256.lower())


class AssessmentRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    # None (default) = no cap on answer length; the UI never sends one.
    max_tokens: int | None = Field(default=None, ge=64)
    audience: Literal["expert", "familiar", "learner"] = "expert"
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
        self.corpus: dict[str, Any] = {
            "status": "pending" if self.settings.auto_ingest_corpus else "disabled",
            "dir": str(self.settings.corpus_dir),
            "files": 0,
            "ingested": 0,
            "already_indexed": 0,
            "failed": [],
        }
        self.corpus_hashes: set[str] = set()
        self.session_uploads: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Lex6Service:
        from lex6.services import build_ingestion_pipeline, build_llm, build_retriever, build_vector_store

        s = settings or get_settings()
        vs = build_vector_store(s)
        llm = build_llm(s)

        def agent_factory(retriever: Retriever, max_tokens: int | None, audience: str = "expert") -> RetrievalAgent:
            return RetrievalAgent(retriever, llm, answer_max_tokens=max_tokens, audience=audience)

        return cls(vs, build_ingestion_pipeline(s, vs), build_retriever("advanced", s, vs), agent_factory, s)

    # --- documents -------------------------------------------------------------------
    def sync_corpus(self) -> dict[str, Any]:
        """Index every file in ``corpus_dir`` that isn't in the persisted store yet (matched by content hash)."""
        root = self.settings.corpus_dir
        if not root.is_dir():
            self.corpus.update(status="missing")
            return self.corpus
        skip = (self.settings.vector_store_dir.resolve(), self.settings.uploads_dir.resolve())
        files = [p for p in self.pipeline.expand([root]) if not any(p.resolve().is_relative_to(d) for d in skip)]
        self.corpus_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.corpus.update(status="syncing", files=len(files), ingested=0, already_indexed=0, failed=[])
        for path in files:
            with self.lock:
                (report,) = self.pipeline.ingest([path], source="corpus")
            if report.status == "ingested":
                self.corpus["ingested"] += 1
            elif any(e.startswith("duplicate") for e in report.errors):
                self.corpus["already_indexed"] += 1
            else:
                self.corpus["failed"].append({"filename": report.filename, "errors": report.errors})
        self.corpus["status"] = "ready"
        logger.info(
            "corpus %s: %d files, %d newly ingested, %d already indexed, %d failed",
            root,
            len(files),
            self.corpus["ingested"],
            self.corpus["already_indexed"],
            len(self.corpus["failed"]),
        )
        return self.corpus

    def documents(self) -> list[dict[str, Any]]:
        return [
            {
                **d,
                "source": "corpus" if d.get("sha256") in self.corpus_hashes else d.get("source", "upload"),
                "session_upload": d["id"] in self.session_uploads,
            }
            for d in self.vector_store.list_documents()
        ]

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
        self.session_uploads.update(r.document_id for r in reports if r.status == "ingested")
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
            response = self.agent_factory(retriever, req.max_tokens, req.audience).run(req.query)
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
    state: dict[str, Lex6Service] = {}
    state_lock = threading.Lock()

    def svc() -> Lex6Service:
        with state_lock:
            if "service" not in state:
                state["service"] = service or Lex6Service.from_settings(settings)
            return state["service"]

    def sync_corpus() -> None:
        s = svc()
        try:
            s.sync_corpus()
        except Exception as e:
            logger.exception("corpus sync failed")
            s.corpus.update(status="error", error=str(e))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        cfg = service.settings if service else (settings or get_settings())
        if cfg.auto_ingest_corpus:
            threading.Thread(target=sync_corpus, name="corpus-sync", daemon=True).start()
        yield

    app = FastAPI(title="Lex6 agentic RAG", lifespan=lifespan)

    cfg = service.settings if service else (settings or get_settings())
    login_required = bool(cfg.access_passcode_sha256)
    access_token = hmac.new(
        (cfg.session_secret or secrets.token_hex(32)).encode(), b"lex6-access", hashlib.sha256
    ).hexdigest()

    def logged_in(request: Request) -> bool:
        return not login_required or hmac.compare_digest(request.cookies.get(ACCESS_COOKIE, ""), access_token)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if request.url.path in PUBLIC_PATHS or logged_in(request):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "login required"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    @app.get("/login", include_in_schema=False)
    def login_page(request: Request) -> Response:
        if login_required and logged_in(request):
            return RedirectResponse("/", status_code=303)
        return FileResponse(STATIC_DIR / "login.html")

    @app.post("/api/login")
    def login(req: LoginRequest, response: Response) -> dict[str, bool]:
        if login_required and not passcode_matches(req.passcode, cfg.access_passcode_sha256):
            logger.warning("rejected login attempt")
            raise HTTPException(401, "incorrect passcode")
        response.set_cookie(ACCESS_COOKIE, access_token, httponly=True, samesite="strict")
        return {"ok": True}

    @app.post("/api/logout")
    def logout(response: Response) -> dict[str, bool]:
        response.delete_cookie(ACCESS_COOKIE)
        return {"ok": True}

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
            "corpus": s.corpus,
        }

    @app.get("/api/documents")
    def list_documents() -> list[dict[str, Any]]:
        return svc().documents()

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
