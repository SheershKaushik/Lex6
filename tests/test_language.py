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


def test_collapse_repetitions_removes_translation_loops():
    from lex6.ingestion.language import collapse_repetitions

    loop = "and Considering that the Convention should be implemented in a manner that is fair, "
    text = "The Parties, " + loop * 5 + "have agreed as follows: Article 1. Article 1."
    out = collapse_repetitions(text)
    assert out.count("Considering that the Convention") == 1
    assert out.startswith("The Parties, and Considering") and out.endswith("Article 1. Article 1.")
    assert collapse_repetitions("A, B. A, B.") == "A, B. A, B."


def test_translator_caps_tokens_and_collapses_output():
    from lex6.ingestion.language import LLMTranslator
    from tests.conftest import ScriptedLLM

    llm = ScriptedLLM(fn=lambda p, s: "It shall be fair, ambitious and effective; " * 4)
    out = LLMTranslator(llm).translate("Il doit être équitable.", "fr")
    assert out == "It shall be fair, ambitious and effective;"
    assert llm.prompts[0]["max_tokens"] == len("Il doit être équitable.") // 2 + 128


def test_translator_retries_degenerate_pieces_in_halves():
    from lex6.ingestion.language import LLMTranslator
    from tests.conftest import ScriptedLLM

    def fn(prompt, system):
        src = prompt.split("\n\n", 1)[1]
        if len(src) > 400:  # long input -> the model loops
            return "and Considering that it is fair, transparent and predictable, " * 6
        return "EN:" + src[:20]

    llm = ScriptedLLM(fn=fn)
    text = ("Considérant que la Convention doit être mise en œuvre de manière équitable. " * 4 + "\n\n") * 2
    out = LLMTranslator(llm, max_chars=1000).translate(text, "fr")
    assert "Considering" not in out
    assert out.count("EN:") >= 2
    assert len(llm.prompts) >= 3
