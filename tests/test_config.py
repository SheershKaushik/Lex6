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


def test_relative_paths_resolve_against_project_root_not_cwd(monkeypatch, tmp_path):
    from lex6.config import PROJECT_ROOT

    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None)
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.corpus_dir == s.data_dir and s.vector_store_dir == PROJECT_ROOT / "data" / "vector_store"
    monkeypatch.setenv("LEX6_CORPUS_DIR", "data/sample")
    assert Settings(_env_file=None).corpus_dir == PROJECT_ROOT / "data" / "sample"
    monkeypatch.setenv("LEX6_DATA_DIR", str(tmp_path / "store"))
    assert Settings(_env_file=None).data_dir == tmp_path / "store"


def test_project_root_override_and_env_file_location(monkeypatch, tmp_path):
    import lex6.config as config

    monkeypatch.setenv("LEX6_HOME", str(tmp_path))
    assert config.find_project_root() == tmp_path.resolve()
    monkeypatch.delenv("LEX6_HOME")
    assert config.find_project_root() == Path(__file__).resolve().parents[1]
    assert Path(Settings.model_config["env_file"]) == config.PROJECT_ROOT / ".env"
