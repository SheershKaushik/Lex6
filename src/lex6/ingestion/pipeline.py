"""End-to-end ingestion: validate -> load -> content check -> translate to English -> chunk -> embed -> store."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lex6.config import Settings, get_settings
from lex6.embeddings import Embedder
from lex6.ingestion.chunking import split_documents
from lex6.ingestion.language import LanguageNormalizer, TranslationError, Translator
from lex6.ingestion.loaders import discover_files, extract_country_from_filename, load_file
from lex6.ingestion.validation import FileValidator, validate_content
from lex6.llm import LLMError
from lex6.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FileReport:
    filename: str
    status: str  # ingested | rejected | failed
    document_id: str | None = None
    file_type: str | None = None
    language: str | None = None
    translated_pages: int = 0
    pages: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class IngestionPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        translator: Translator | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.vector_store = vector_store
        self.normalizer = LanguageNormalizer(translator)

    def expand(self, paths: list[str | Path]) -> list[Path]:
        files: list[Path] = []
        for p in map(Path, paths):
            files.extend(discover_files(p) if p.is_dir() else [p])
        return files

    def ingest(
        self, paths: list[str | Path], display_names: dict[Path, str] | None = None, source: str = "upload"
    ) -> list[FileReport]:
        validator = FileValidator(self.settings.max_file_size_mb, known_hashes=self.vector_store.known_hashes())
        reports = []
        changed = False
        for path in self.expand(paths):
            name = (display_names or {}).get(path, path.name)
            report = self._ingest_one(path, name, validator, source)
            reports.append(report)
            changed |= report.status == "ingested"
            logger.info("%s: %s %s", name, report.status, "; ".join(report.errors))
        if changed:
            self.vector_store.save_local()
        return reports

    def _ingest_one(self, path: Path, name: str, validator: FileValidator, source: str = "upload") -> FileReport:
        v = validator.validate(path)
        report = FileReport(filename=name, status="rejected", file_type=v.file_type, warnings=list(v.warnings))
        if not v.ok:
            report.errors = v.errors
            return report

        try:
            pages = load_file(path, v.file_type)
        except Exception as e:
            report.errors.append(f"could not load file: {e}")
            return report
        for p in pages:
            p.metadata["source_file"] = name
        content_errors = validate_content(pages, self.settings.min_text_chars)
        if content_errors:
            report.errors = content_errors
            return report

        try:
            pages, lang_summary = self.normalizer.normalize(pages)
        except (TranslationError, LLMError) as e:
            report.status = "failed"
            report.errors.append(str(e))
            return report
        info = lang_summary.get(name, {})
        report.language = info.get("language")
        report.translated_pages = info.get("translated_pages", 0)
        report.pages = len(pages)

        document_id = v.sha256[:16]
        country = extract_country_from_filename(name)
        for p in pages:
            p.metadata.update(document_id=document_id, country=country)

        chunks = split_documents(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            report.errors.append("no chunks produced")
            return report
        try:
            embeddings = self.embedder.generate_embeddings([c.page_content for c in chunks])
            self.vector_store.add_documents(chunks, embeddings)
        except Exception as e:
            report.status = "failed"
            report.errors.append(f"embedding/storage failed: {e}")
            return report

        record = {
            "id": document_id,
            "title": name,
            "filename": name,
            "doc_type": v.file_type,
            "sha256": v.sha256,
            "size_bytes": v.size_bytes,
            "language": report.language,
            "translated_pages": report.translated_pages,
            "pages": len(pages),
            "chunks": len(chunks),
            "country": country,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        page_records = [
            {
                "page_number": int(p.metadata.get("page", i)) + 1,
                "text": p.page_content,
                "original_language": p.metadata.get("original_language"),
                "translated": p.metadata.get("translated", False),
            }
            for i, p in enumerate(pages)
        ]
        self.vector_store.register_document(record, page_records)
        validator.known_hashes.add(v.sha256)
        report.status, report.document_id, report.chunks = "ingested", document_id, len(chunks)
        return report
