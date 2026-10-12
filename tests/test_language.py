import pytest
from langchain_core.documents import Document

from lex6.ingestion.language import (
    LanguageNormalizer,
    LLMTranslator,
    TranslationError,
    detect_language,
    language_name,
)
from tests.conftest import FakeTranslator, ScriptedLLM

EN = "Each Party shall prepare, communicate and maintain successive nationally determined contributions."
FR = "Chaque Partie établit, communique et actualise les contributions déterminées au niveau national successives."


def test_detect_language():
    assert detect_language(EN) == "en"
    assert detect_language(FR) == "fr"
    assert detect_language("   ") is None
    assert language_name("fr") == "French"
    assert language_name("zh-cn") == "Chinese"


def test_llm_translator_splits_long_text():
    llm = ScriptedLLM(fn=lambda p, s: "translated")
    text = "\n\n".join([FR] * 30)
    out = LLMTranslator(llm, max_chars=300).translate(text, "fr")
    assert len(llm.prompts) > 1
    assert all("French" in p["prompt"] for p in llm.prompts)
    assert out == "\n".join(["translated"] * len(llm.prompts))


def test_llm_translator_empty_output_raises():
    with pytest.raises(TranslationError):
        LLMTranslator(ScriptedLLM(responses=[""])).translate(FR, "fr")


def test_normalizer_translates_only_non_english_pages():
    docs = [
        Document(page_content=EN, metadata={"source_file": "mixed.pdf", "page": 0}),
        Document(page_content=FR, metadata={"source_file": "mixed.pdf", "page": 1}),
    ]
    tr = FakeTranslator()
    out, summary = LanguageNormalizer(tr).normalize(docs)
    assert out[0].page_content == EN and out[0].metadata["translated"] is False
    assert out[1].page_content.startswith("[translated from fr]")
    assert out[1].metadata == {
        "source_file": "mixed.pdf",
        "page": 1,
        "original_language": "fr",
        "translated": True,
    }
    assert summary["mixed.pdf"]["translated_pages"] == 1
    assert tr.calls == ["fr"]


def test_short_page_inherits_document_language():
    docs = [
        Document(page_content=FR * 3, metadata={"source_file": "fr.pdf"}),
        Document(page_content="Article 2", metadata={"source_file": "fr.pdf"}),
    ]
    out, _ = LanguageNormalizer(FakeTranslator()).normalize(docs)
    assert all(d.metadata["original_language"] == "fr" for d in out)


def test_non_english_without_translator_fails():
    with pytest.raises(TranslationError, match="no translator"):
        LanguageNormalizer(None).normalize([Document(page_content=FR, metadata={"source_file": "f.txt"})])


def test_translator_exception_wrapped():
    class Boom:
        def translate(self, text, lang):
            raise ConnectionError("ollama down")

    with pytest.raises(TranslationError, match="ollama down"):
        LanguageNormalizer(Boom()).normalize([Document(page_content=FR, metadata={"source_file": "f.txt"})])
