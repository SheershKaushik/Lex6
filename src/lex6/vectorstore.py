"""FAISS vector store with a persisted docstore (text, metadata, document registry)."""

from __future__ import annotations

import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissDocStore:
    """Text + metadata for each vector keyed by FAISS internal index, plus a per-file registry."""

    def __init__(self):
        self.store: dict[int, dict[str, Any]] = {}
        self.next_index = 0
        self.documents: dict[str, dict[str, Any]] = {}  # document_id -> registry record
        self.pages: dict[str, list[dict[str, Any]]] = {}  # document_id -> [{page_number, text}]
        self.info: dict[str, Any] = {}  # embedding_model, embedding_dim

    def add(self, text: str, metadata: dict[str, Any], original_id: str) -> int:
        idx = self.next_index
        self.store[idx] = {"text": text, "metadata": metadata, "original_id": original_id}
        self.next_index += 1
        return idx

    def get_by_index(self, index: int) -> dict[str, Any] | None:
        return self.store.get(int(index))

    def count(self) -> int:
        return len(self.store)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "store": self.store,
                    "next_index": self.next_index,
                    "documents": self.documents,
                    "pages": self.pages,
                    "info": self.info,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> FaissDocStore:
        instance = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.store = data["store"]
        instance.next_index = data["next_index"]
        instance.documents = data.get("documents", {})
        instance.pages = data.get("pages", {})
        instance.info = data.get("info", {})
        return instance


class VectorStoreMismatchError(RuntimeError):
    pass


class VectorStore:
    """FAISS ``IndexFlatL2`` over L2-normalised vectors.

    ``IndexFlatL2`` returns *squared* L2 distances, so ``cosine = 1 - distance / 2``.
    """

    def __init__(
        self,
        embedding_dim: int,
        collection_name: str = "pdf_documents",
        persist_directory: str | Path = "data/vector_store",
        embedding_model: str | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.index: faiss.Index = faiss.IndexFlatL2(embedding_dim)
        self.docstore = FaissDocStore()
        self.docstore.info = {"embedding_model": embedding_model, "embedding_dim": embedding_dim}
        if self._exists_on_disk():
            self.load_local()

    @property
    def _index_path(self) -> Path:
        return self.persist_directory / f"{self.collection_name}.faiss"

    @property
    def _docstore_path(self) -> Path:
        return self.persist_directory / f"{self.collection_name}_docstore.pkl"

    def _exists_on_disk(self) -> bool:
        return self._index_path.exists() and self._docstore_path.exists()

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add_documents(self, documents: list[Any], embeddings: np.ndarray) -> list[str]:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        if len(documents) == 0:
            return []
        vectors = np.ascontiguousarray(embeddings, dtype="float32").copy()
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Embeddings must have shape (n, {self.embedding_dim}), got {vectors.shape}")
        faiss.normalize_L2(vectors)

        start = self.index.ntotal
        ids = []
        self.index.add(vectors)
        for i, doc in enumerate(documents):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{start + i}"
            metadata = dict(doc.metadata)
            metadata["content_length"] = len(doc.page_content)
            self.docstore.add(doc.page_content, metadata, doc_id)
            ids.append(doc_id)
        if self.index.ntotal != self.docstore.count():
            raise RuntimeError("Faiss index count and DocStore count mismatch after addition")
        return ids

    def search_embeddings(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Return 1-D ``(squared_l2_distances, indices)`` for a single ``(1, dim)`` query."""
        if self.index.ntotal == 0 or top_k <= 0:
            return np.array([], dtype="float32"), np.array([], dtype="int64")
        q = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1).copy()
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))
        return distances[0], indices[0]

    def all_chunks(self) -> list[dict[str, Any]]:
        return [self.docstore.store[i] for i in sorted(self.docstore.store)]

    # --- document registry -------------------------------------------------------------
    def register_document(self, record: dict[str, Any], pages: list[dict[str, Any]] | None = None) -> None:
        self.docstore.documents[record["id"]] = record
        if pages is not None:
            self.docstore.pages[record["id"]] = pages

    def list_documents(self) -> list[dict[str, Any]]:
        return sorted(self.docstore.documents.values(), key=lambda d: d.get("uploaded_at", ""))

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        return self.docstore.documents.get(document_id)

    def get_pages(self, document_id: str) -> list[dict[str, Any]]:
        return self.docstore.pages.get(document_id, [])

    def known_hashes(self) -> set[str]:
        return {d["sha256"] for d in self.docstore.documents.values() if d.get("sha256")}

    # --- persistence -------------------------------------------------------------------
    def save_local(self) -> None:
        os.makedirs(self.persist_directory, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))
        self.docstore.save(self._docstore_path)
        logger.info("Vector store saved to %s (%d vectors)", self.persist_directory, self.index.ntotal)

    def load_local(self) -> None:
        index = faiss.read_index(str(self._index_path))
        docstore = FaissDocStore.load(self._docstore_path)
        stored_model = docstore.info.get("embedding_model")
        if index.d != self.embedding_dim or (
            self.embedding_model and stored_model and stored_model != self.embedding_model
        ):
            raise VectorStoreMismatchError(
                f"Persisted store at {self.persist_directory} was built with {stored_model} (dim {index.d}); "
                f"current embedder is {self.embedding_model} (dim {self.embedding_dim}). "
                "Delete the directory or point LEX6_DATA_DIR elsewhere and re-ingest."
            )
        if index.ntotal != docstore.count():
            raise VectorStoreMismatchError("Persisted index and docstore have different sizes")
        self.index, self.docstore = index, docstore
        self.docstore.info = {
            "embedding_model": self.embedding_model or stored_model,
            "embedding_dim": index.d,
        }
        logger.info("Vector store loaded from %s (%d vectors)", self.persist_directory, index.ntotal)
