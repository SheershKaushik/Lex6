"""Document loaders for PDF (PyMuPDF), DOCX and plain-text files."""

from __future__ import annotations

import re
from pathlib import Path

import pycountry
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from lex6.ingestion.validation import SUPPORTED_EXTENSIONS


def discover_files(directory: str | Path) -> list[Path]:
    """Recursively find all supported files in ``directory``."""
    root = Path(directory)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_file(path: str | Path, file_type: str) -> list[Document]:
    """Load a file into LangChain ``Document`` pages with ``source_file``/``file_type``/``page`` metadata."""
    path = Path(path)
    if file_type == "pdf":
        docs = PyMuPDFLoader(str(path)).load()
    elif file_type == "docx":
        docs = Docx2txtLoader(str(path)).load()
    elif file_type == "text":
        docs = TextLoader(str(path), autodetect_encoding=True).load()
    else:
        raise ValueError(f"unsupported file type: {file_type}")

    for i, doc in enumerate(docs):
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = file_type
        doc.metadata["page"] = int(doc.metadata.get("page", i))
    return docs


def _build_country_name_list() -> list[str]:
    names = set()
    for country in pycountry.countries:
        for attr in ("name", "common_name", "official_name"):
            if hasattr(country, attr):
                names.add(getattr(country, attr))
    return sorted(names, key=len, reverse=True)


COUNTRY_NAMES = _build_country_name_list()


def extract_country_from_filename(filename: str) -> str | None:
    """Match a known country name in a filename (longest first, whole words only)."""
    clean = re.sub(r"[_\-.]+", " ", Path(filename).stem).lower()
    for country in COUNTRY_NAMES:
        if re.search(rf"\b{re.escape(country.lower())}\b", clean):
            return country
    return None
