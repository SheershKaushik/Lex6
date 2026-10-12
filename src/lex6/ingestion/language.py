"""Language detection and translation to English (run during validation, before chunking)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

import pycountry
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import DetectorFactory, LangDetectException, detect

from lex6.llm import LLM

DetectorFactory.seed = 0  # reproducible langdetect output
logger = logging.getLogger(__name__)

ENGLISH = "en"


class TranslationError(RuntimeError):
    pass


def language_name(code: str | None) -> str:
    if not code:
        return "unknown"
    base = code.split("-")[0]
    lang = pycountry.languages.get(alpha_2=base)
    return lang.name if lang else code


def detect_language(text: str, sample_chars: int = 2000) -> str | None:
    """Detect the ISO 639-1 language of ``text`` from its first ``sample_chars`` characters."""
    sample = (text or "")[:sample_chars].strip()
    if not sample:
        return None
    try:
        return detect(sample)
    except LangDetectException:
        return None


class Translator(Protocol):
    def translate(self, text: str, source_language: str) -> str: ...


TRANSLATION_SYSTEM = (
    "You are a professional legal translator. Translate the user's text into English. "
    "Preserve article numbers, paragraph numbering, defined terms and line structure. "
    "Output ONLY the English translation, with no preamble, notes or commentary."
)


class LLMTranslator:
    """Translates text with a local Ollama model, in pieces small enough for the model's context."""

    def __init__(self, llm: LLM, max_chars: int = 1500):
        self.llm = llm
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=0)

    def translate(self, text: str, source_language: str) -> str:
        if not text.strip():
            return text
        pieces = self.splitter.split_text(text)
        out = []
        for piece in pieces:
            prompt = f"Translate this {language_name(source_language)} text into English:\n\n{piece}"
            translated = self.llm.generate(prompt, system=TRANSLATION_SYSTEM, max_tokens=2048).strip()
            if not translated:
                raise TranslationError("translator returned empty output")
            out.append(translated)
        return "\n".join(out)


class LanguageNormalizer:
    """Ensures every page is English: detects language per page and translates non-English pages.

    Short pages (fewer than ``min_detect_chars``) inherit the language detected for the whole file,
    since langdetect is unreliable on very short strings.
    """

    def __init__(self, translator: Translator | None, min_detect_chars: int = 50):
        self.translator = translator
        self.min_detect_chars = min_detect_chars

    def normalize(self, documents: list[Document]) -> tuple[list[Document], dict[str, dict]]:
        by_source: dict[str, list[Document]] = defaultdict(list)
        for d in documents:
            by_source[d.metadata.get("source_file", "unknown")].append(d)

        summary: dict[str, dict] = {}
        result: list[Document] = []
        for source, pages in by_source.items():
            doc_lang = detect_language(" ".join(p.page_content for p in pages))
            translated_pages = 0
            for page in pages:
                text = page.page_content
                lang = detect_language(text) if len(text.strip()) >= self.min_detect_chars else doc_lang
                lang = lang or doc_lang
                meta = dict(page.metadata, original_language=lang or "unknown", translated=False)
                if lang and lang != ENGLISH and text.strip():
                    if self.translator is None:
                        raise TranslationError(
                            f"{source}: {language_name(lang)} content found but no translator is configured"
                        )
                    try:
                        text = self.translator.translate(text, lang)
                    except TranslationError:
                        raise
                    except Exception as e:
                        raise TranslationError(f"{source}: translation from {language_name(lang)} failed: {e}") from e
                    meta["translated"] = True
                    translated_pages += 1
                result.append(Document(page_content=text, metadata=meta))
            summary[source] = {
                "language": doc_lang or "unknown",
                "language_name": language_name(doc_lang),
                "translated_pages": translated_pages,
                "pages": len(pages),
            }
            logger.info("%s: language=%s, translated %d/%d pages", source, doc_lang, translated_pages, len(pages))
        return result, summary
