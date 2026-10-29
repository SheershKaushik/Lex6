"""LLM wrappers around Ollama (local by default, Ollama Cloud for HyDE when a key is available)."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from lex6.config import Settings, get_settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove reasoning blocks emitted by reasoning models such as deepseek-r1."""
    text = _THINK_RE.sub("", text or "")
    # An unterminated <think> (e.g. truncated output) leaves only reasoning; drop it.
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return text.strip()


class LLMError(RuntimeError):
    """An Ollama call failed; the message says how to fix it (pull the model, start the server, ...)."""


def model_missing_message(model: str, host: str) -> str:
    return (
        f"Ollama model '{model}' is not installed on {host}. Run `ollama pull {model}`, "
        "or set LEX6_LLM_MODEL to an installed model (see `ollama list`)."
    )


def explain_ollama_error(exc: Exception, model: str, host: str) -> LLMError:
    """Turn ``ollama.ResponseError`` / connection failures into an actionable ``LLMError``."""
    status = getattr(exc, "status_code", None)
    detail = getattr(exc, "error", None) or str(exc)
    if status == 404:
        msg = model_missing_message(model, host)
    elif status in (401, 403):
        msg = f"Ollama rejected the request for '{model}' on {host} ({status}): check OLLAMA_API_KEY. {detail}"
    elif status is not None:
        msg = f"Ollama error {status} from model '{model}' on {host}: {detail}"
    else:
        msg = f"Cannot reach Ollama at {host} ({detail}). Start it with `ollama serve`."
    return LLMError(msg)


_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (ConnectionError,)
try:
    import httpx

    _CONNECTION_ERRORS += (httpx.TransportError,)
except ImportError:  # pragma: no cover
    pass


def _ollama_errors() -> tuple[type[BaseException], ...]:
    from ollama import ResponseError

    return (ResponseError, *_CONNECTION_ERRORS)


def check_model(model: str, host: str) -> str | None:
    """Return ``None`` if ``model`` is installed on the Ollama server at ``host``, else a problem description."""
    from ollama import Client

    try:
        names = {m.model for m in Client(host=host).list().models}
    except Exception as e:
        return str(explain_ollama_error(e, model, host))
    if model in names or f"{model}:latest" in names:
        return None
    return model_missing_message(model, host)


class LLM(Protocol):
    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str: ...


class OllamaLLM:
    """Chat model served by a local Ollama server (ChatOllama, as in the notebook)."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        settings: Settings | None = None,
    ):
        s = settings or get_settings()
        self.model_name = model_name or s.llm_model
        self.base_url = base_url or s.ollama_base_url
        self.temperature = s.llm_temperature if temperature is None else temperature
        self.num_predict = num_predict or s.llm_num_predict
        self.num_ctx = s.llm_num_ctx
        self.reasoning_budget = s.llm_reasoning_budget
        self._thinks: bool | None = None
        self._clients: dict[int, object] = {}

    def supports_thinking(self) -> bool:
        """Whether Ollama reports the ``thinking`` capability (deepseek-r1, qwen3, ...) for this model."""
        if self._thinks is None:
            from ollama import Client

            try:
                caps = Client(host=self.base_url).show(self.model_name).capabilities or []
            except Exception as e:  # server down / model missing: generate() reports it properly
                logger.debug("could not read capabilities of %s: %s", self.model_name, e)
                return False
            self._thinks = "thinking" in caps
        return self._thinks

    def token_limit(self, max_tokens: int | None) -> int:
        """num_predict for a call: uncapped answers stay uncapped; capped calls get room for reasoning."""
        if not max_tokens:
            return self.num_predict
        return max_tokens + self.reasoning_budget if self.supports_thinking() else max_tokens

    def _client(self, num_predict: int):
        if num_predict not in self._clients:
            from langchain_ollama import ChatOllama

            self._clients[num_predict] = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=num_predict,
                num_ctx=self.num_ctx,
            )
        return self._clients[num_predict]

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        limit = self.token_limit(max_tokens)
        try:
            response = self._client(limit).invoke(messages)
        except _ollama_errors() as e:
            raise explain_ollama_error(e, self.model_name, self.base_url) from e
        if (getattr(response, "response_metadata", None) or {}).get("done_reason") == "length":
            if max_tokens:
                hint = "raise LEX6_LLM_REASONING_BUDGET" if self.supports_thinking() else "runaway output was capped"
                logger.warning(
                    "%s output hit its per-call cap (num_predict=%s for a %s-token task); %s",
                    self.model_name,
                    limit,
                    max_tokens,
                    hint,
                )
            else:
                logger.warning(
                    "%s output was cut off by the context window (num_ctx=%s); raise LEX6_LLM_NUM_CTX",
                    self.model_name,
                    self.num_ctx,
                )
        return strip_think(response.content)


class LLMClientWrapper:
    """Raw ``ollama.Client.generate`` wrapper used for HyDE (notebook: Ollama Cloud ``gpt-oss:120b-cloud``).

    Cloud models (``*-cloud``) are only used when ``OLLAMA_API_KEY`` is set. Otherwise the wrapper
    falls back to the local server and the local chat model so the pipeline still runs offline.
    """

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        settings: Settings | None = None,
    ):
        from ollama import Client

        s = settings or get_settings()
        api_key = api_key or s.ollama_api_key
        model_name = model_name or s.hyde_model
        if model_name.endswith("-cloud") and not api_key:
            logger.info("No OLLAMA_API_KEY: using local model %s for HyDE instead of %s", s.llm_model, model_name)
            model_name, host = s.llm_model, host or s.ollama_base_url
        elif model_name.endswith("-cloud"):
            host = host or s.ollama_cloud_host
        else:
            host = host or s.ollama_base_url
        self.model_name = model_name
        self.host = host
        headers = {"Authorization": f"Bearer {api_key}"} if api_key and host == s.ollama_cloud_host else {}
        self.client = Client(host=host, headers=headers)

    def generate_text(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
        except _ollama_errors() as e:
            raise explain_ollama_error(e, self.model_name, self.host) from e
        return strip_think(response["response"])

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        return self.generate_text(full, max_tokens=max_tokens or 300)
