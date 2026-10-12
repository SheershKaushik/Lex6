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
        self._clients: dict[int, object] = {}

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
        response = self._client(max_tokens or self.num_predict).invoke(messages)
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
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return strip_think(response["response"])

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        return self.generate_text(full, max_tokens=max_tokens or 300)
