import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from lex6.config import Settings

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (no model download)."""

    model_name = "fake-bow"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def get_embedding_dim(self) -> int:
        return self.dim

    def generate_embeddings(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text, source_language):
        self.calls.append(source_language)
        return f"[translated from {source_language}] Paris Agreement climate change text."


class ScriptedLLM:
    """Returns queued responses (or a function of the prompt) and records prompts."""

    def __init__(self, responses=None, fn=None):
        self.responses = list(responses or [])
        self.fn = fn
        self.prompts = []

    def generate(self, prompt, *, system=None, max_tokens=None):
        self.prompts.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        if self.fn:
            return self.fn(prompt, system)
        return self.responses.pop(0) if self.responses else ""


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", embedding_model="fake-bow")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_dir():
    return SAMPLE_DIR
