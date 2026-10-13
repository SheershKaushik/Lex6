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


def test_ollama_errors_become_actionable_llm_errors(capsys, monkeypatch):
    import pytest
    from ollama import ResponseError

    from lex6 import cli
    from lex6.llm import LLMError, OllamaLLM, explain_ollama_error

    s = Settings(_env_file=None)
    llm = OllamaLLM(model_name="deepseek-r1:8b", settings=s)

    class Missing:
        def invoke(self, messages):
            raise ResponseError("model 'deepseek-r1:8b' not found", 404)

    llm._clients[64] = Missing()
    with pytest.raises(LLMError, match=r"ollama pull deepseek-r1:8b"):
        llm.generate("hi", max_tokens=64)

    down = OllamaLLM(model_name="qwen2.5:3b", base_url="http://127.0.0.1:1", settings=s)
    with pytest.raises(LLMError, match="ollama serve"):
        down.generate("hi", max_tokens=8)

    assert "OLLAMA_API_KEY" in str(explain_ollama_error(ResponseError("unauthorized", 401), "m", "h"))
    assert "error 500" in str(explain_ollama_error(ResponseError("out of memory", 500), "m", "h"))

    def boom(args):
        raise LLMError("Ollama model 'x' is not installed")

    monkeypatch.setattr(cli, "_cmd_ask", boom)
    assert cli.main(["ask", "q"]) == 1
    assert "not installed" in capsys.readouterr().err


def test_check_model(monkeypatch):
    import ollama

    from lex6.llm import check_model

    class FakeClient:
        def __init__(self, host):
            pass

        def list(self):
            from types import SimpleNamespace

            return SimpleNamespace(models=[SimpleNamespace(model="qwen2.5:3b"), SimpleNamespace(model="llama3:latest")])

    monkeypatch.setattr(ollama, "Client", FakeClient)
    assert check_model("qwen2.5:3b", "h") is None
    assert check_model("llama3", "h") is None
    assert "ollama pull deepseek-r1:8b" in check_model("deepseek-r1:8b", "h")
