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
    llm._thinks = False

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


def test_answers_are_uncapped_and_cutoffs_logged(caplog):
    from types import SimpleNamespace

    from lex6.llm import OllamaLLM

    s = Settings(_env_file=None)
    assert s.llm_num_predict == -1 and s.llm_num_ctx >= 16384
    llm = OllamaLLM(model_name="m", settings=s)
    replies = [
        SimpleNamespace(content="Full answer.", response_metadata={"done_reason": "stop"}),
        SimpleNamespace(content="Cut off mid", response_metadata={"done_reason": "length"}),
    ]

    class Fake:
        def invoke(self, messages):
            return replies.pop(0)

    llm._clients[-1] = Fake()
    with caplog.at_level("WARNING", logger="lex6.llm"):
        assert llm.generate("q") == "Full answer."
        assert not caplog.records
        assert llm.generate("q") == "Cut off mid"
    assert "LEX6_LLM_NUM_CTX" in caplog.text


def test_capped_calls_leave_room_for_reasoning(monkeypatch, caplog):
    from types import SimpleNamespace

    import ollama

    from lex6.llm import OllamaLLM

    shows = []

    class FakeClient:
        def __init__(self, host):
            pass

        def show(self, model):
            shows.append(model)
            caps = ["completion", "thinking"] if model.startswith("deepseek-r1") else ["completion"]
            return SimpleNamespace(capabilities=caps)

    monkeypatch.setattr(ollama, "Client", FakeClient)
    s = Settings(_env_file=None)
    reasoner = OllamaLLM(model_name="deepseek-r1:8b", settings=s)
    plain = OllamaLLM(model_name="qwen2.5:3b", settings=s)
    assert reasoner.token_limit(455) == 455 + s.llm_reasoning_budget
    assert reasoner.token_limit(None) == -1
    assert plain.token_limit(455) == 455 and plain.token_limit(None) == -1
    reasoner.token_limit(10)
    assert shows == ["deepseek-r1:8b", "qwen2.5:3b"]

    class Fake:
        def invoke(self, messages):
            return SimpleNamespace(content="<think>long reasoning", response_metadata={"done_reason": "length"})

    reasoner._clients[455 + s.llm_reasoning_budget] = Fake()
    with caplog.at_level("WARNING", logger="lex6.llm"):
        assert reasoner.generate("q", max_tokens=455) == ""
    assert "LEX6_LLM_REASONING_BUDGET" in caplog.text and "NUM_CTX" not in caplog.text

    class Down:
        def __init__(self, host):
            raise ConnectionError("down")

    monkeypatch.setattr(ollama, "Client", Down)
    assert OllamaLLM(model_name="deepseek-r1:8b", settings=s).token_limit(64) == 64
