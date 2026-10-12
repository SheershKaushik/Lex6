from pathlib import Path

from lex6.config import Settings


def test_defaults_match_reference_notebook():
    s = Settings(_env_file=None)
    assert s.llm_model == "deepseek-r1:8b"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.hyde_model == "gpt-oss:120b-cloud"
    assert (s.chunk_size, s.chunk_overlap) == (1000, 200)


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LEX6_LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("LEX6_DATA_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.llm_model == "qwen2.5:3b"
    assert s.effective_translation_model == "qwen2.5:3b"
    assert s.vector_store_dir == Path(tmp_path) / "vector_store"
