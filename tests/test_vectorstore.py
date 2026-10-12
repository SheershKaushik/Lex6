import numpy as np
import pytest
from langchain_core.documents import Document

from lex6.vectorstore import VectorStore, VectorStoreMismatchError


def _docs(n):
    return [Document(page_content=f"text {i}", metadata={"i": i}) for i in range(n)]


def test_add_search_normalises_vectors(tmp_path):
    vs = VectorStore(3, persist_directory=tmp_path)
    emb = np.array([[10, 0, 0], [0, 5, 0], [0, 0, 2]], dtype="float32")  # deliberately not unit length
    ids = vs.add_documents(_docs(3), emb)
    assert len(ids) == 3 and vs.ntotal == 3
    d, i = vs.search_embeddings(np.array([[3, 0.1, 0]], dtype="float32"), top_k=10)
    assert len(i) == 3  # top_k clamped to ntotal
    assert i[0] == 0
    cos = 1 - d[0] / 2  # IndexFlatL2 returns *squared* L2 distances
    assert cos == pytest.approx(3 / np.sqrt(9.01), rel=1e-4)
    assert vs.docstore.get_by_index(i[0])["metadata"]["content_length"] == len("text 0")


def test_validation_errors(tmp_path):
    vs = VectorStore(3, persist_directory=tmp_path)
    with pytest.raises(ValueError):
        vs.add_documents(_docs(2), np.zeros((1, 3), dtype="float32"))
    with pytest.raises(ValueError):
        vs.add_documents(_docs(1), np.zeros((1, 4), dtype="float32"))
    assert vs.search_embeddings(np.zeros((1, 3), dtype="float32"))[1].size == 0


def test_persist_and_reload(tmp_path):
    vs = VectorStore(3, persist_directory=tmp_path, embedding_model="m")
    vs.add_documents(_docs(2), np.eye(3, dtype="float32")[:2])
    vs.register_document({"id": "d1", "sha256": "abc", "uploaded_at": "2024"}, [{"page_number": 1, "text": "t"}])
    vs.save_local()
    vs2 = VectorStore(3, persist_directory=tmp_path, embedding_model="m")
    assert vs2.ntotal == 2
    assert vs2.known_hashes() == {"abc"}
    assert vs2.get_pages("d1") == [{"page_number": 1, "text": "t"}]
    assert [c["text"] for c in vs2.all_chunks()] == ["text 0", "text 1"]


def test_reload_with_different_model_fails(tmp_path):
    vs = VectorStore(3, persist_directory=tmp_path, embedding_model="m1")
    vs.add_documents(_docs(1), np.eye(3, dtype="float32")[:1])
    vs.save_local()
    with pytest.raises(VectorStoreMismatchError):
        VectorStore(3, persist_directory=tmp_path, embedding_model="m2")
    with pytest.raises(VectorStoreMismatchError):
        VectorStore(4, persist_directory=tmp_path, embedding_model="m1")
