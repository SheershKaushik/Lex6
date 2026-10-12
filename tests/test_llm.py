from lex6.config import Settings
from lex6.llm import LLMClientWrapper, strip_think


def test_strip_think():
    assert strip_think("<think>reasoning\nmore</think>\nAnswer") == "Answer"
    assert strip_think("Answer only") == "Answer only"
    assert strip_think("Partial <think>never closed") == "Partial"


def test_hyde_falls_back_to_local_without_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    s = Settings(_env_file=None, llm_model="qwen2.5:3b")
    w = LLMClientWrapper(settings=s)
    assert w.model_name == "qwen2.5:3b" and w.host == s.ollama_base_url


def test_hyde_uses_cloud_with_key(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    s = Settings(_env_file=None)
    w = LLMClientWrapper(settings=s)
    assert w.model_name == "gpt-oss:120b-cloud" and w.host == "https://ollama.com"
