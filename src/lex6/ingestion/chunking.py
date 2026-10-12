"""Chunking (RecursiveCharacterTextSplitter, 1000/200 as in the notebook)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source_file", "unknown")
        c.metadata["chunk_index"] = counters.get(src, 0)
        counters[src] = c.metadata["chunk_index"] + 1
    return chunks
