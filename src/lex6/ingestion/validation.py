"""File and content validation run before any document enters the pipeline."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text",
    ".md": "text",
}

_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "cp1252")


@dataclass
class ValidationResult:
    path: Path
    file_type: str | None = None
    sha256: str | None = None
    size_bytes: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FileValidator:
    """Checks that a file is a real, readable, non-duplicate PDF/DOCX/text file within the size limit."""

    def __init__(self, max_file_size_mb: float = 50.0, known_hashes: set[str] | None = None):
        self.max_bytes = int(max_file_size_mb * 1024 * 1024)
        self.known_hashes = known_hashes if known_hashes is not None else set()

    def validate(self, path: str | Path) -> ValidationResult:
        path = Path(path)
        result = ValidationResult(path=path)

        if not path.exists() or not path.is_file():
            result.errors.append("file does not exist")
            return result

        result.file_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
        if result.file_type is None:
            result.errors.append(
                f"unsupported file type '{path.suffix}'; supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return result

        result.size_bytes = path.stat().st_size
        if result.size_bytes == 0:
            result.errors.append("file is empty")
            return result
        if result.size_bytes > self.max_bytes:
            result.errors.append(
                f"file is {result.size_bytes / 1048576:.1f} MB, above the {self.max_bytes / 1048576:.0f} MB limit"
            )
            return result

        check = {"pdf": self._check_pdf, "docx": self._check_docx, "text": self._check_text}[result.file_type]
        check(path, result)
        if not result.ok:
            return result

        result.sha256 = sha256_of(path)
        if result.sha256 in self.known_hashes:
            result.errors.append("duplicate: identical file already ingested")
        return result

    @staticmethod
    def _check_pdf(path: Path, result: ValidationResult) -> None:
        with open(path, "rb") as f:
            if not f.read(1024).lstrip().startswith(b"%PDF-"):
                result.errors.append("not a valid PDF (missing %PDF header)")
                return
        import pymupdf

        try:
            with pymupdf.open(path) as doc:
                if doc.needs_pass:
                    result.errors.append("PDF is password protected")
                elif doc.page_count == 0:
                    result.errors.append("PDF has no pages")
        except Exception as e:  # corrupted file
            result.errors.append(f"PDF could not be opened: {e}")

    @staticmethod
    def _check_docx(path: Path, result: ValidationResult) -> None:
        if not zipfile.is_zipfile(path):
            result.errors.append("not a valid DOCX (not a zip container)")
            return
        try:
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" not in z.namelist():
                    result.errors.append("not a valid DOCX (word/document.xml missing)")
        except zipfile.BadZipFile as e:
            result.errors.append(f"DOCX could not be opened: {e}")

    @staticmethod
    def _check_text(path: Path, result: ValidationResult) -> None:
        raw = path.read_bytes()
        if b"\x00" in raw and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            result.errors.append("text file contains binary data")
            return
        for enc in _TEXT_ENCODINGS:
            try:
                raw.decode(enc)
                if enc not in ("utf-8", "utf-8-sig"):
                    result.warnings.append(f"text decoded as {enc}")
                return
            except UnicodeDecodeError:
                continue
        result.errors.append("text file is not valid UTF-8/UTF-16/cp1252")


def validate_content(documents: list[Document], min_text_chars: int = 20) -> list[str]:
    """Post-load validation: the file must yield extractable text (e.g. not a scanned-only PDF)."""
    errors: list[str] = []
    if not documents:
        errors.append("no content could be loaded")
        return errors
    total = sum(len(d.page_content.strip()) for d in documents)
    if total < min_text_chars:
        errors.append(f"only {total} characters of extractable text (scanned/image-only documents are not supported)")
    return errors
