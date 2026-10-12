import shutil

from lex6.ingestion.pipeline import IngestionPipeline
from lex6.vectorstore import VectorStore
from tests.conftest import FakeTranslator


def _pipeline(settings, embedder, translator=None):
    vs = VectorStore(embedder.get_embedding_dim(), persist_directory=settings.vector_store_dir, embedding_model="fake")
    return IngestionPipeline(embedder, vs, translator or FakeTranslator(), settings), vs


def test_ingest_sample_corpus(settings, fake_embedder, sample_dir, tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(sample_dir, corpus)
    (corpus / "broken.pdf").write_text("not a pdf")
    (corpus / "empty.txt").write_text("")

    translator = FakeTranslator()
    pipe, vs = _pipeline(settings, fake_embedder, translator)
    reports = {r.filename: r for r in pipe.ingest([corpus])}

    assert reports["broken.pdf"].status == "rejected"
    assert reports["empty.txt"].status == "rejected"
    for name in [
        "Paris_agreement-English.pdf",
        "Accord_de_Paris-extrait-FR.pdf",
        "Sample_bilateral_agreement_Article6.docx",
        "Article6_glossary.txt",
    ]:
        assert reports[name].status == "ingested", reports[name].errors
        assert reports[name].chunks > 0

    fr = reports["Accord_de_Paris-extrait-FR.pdf"]
    assert fr.language == "fr" and fr.translated_pages == fr.pages == 2
    assert reports["Paris_agreement-English.pdf"].translated_pages == 0
    assert set(translator.calls) == {"fr"}

    # every stored chunk is English and carries provenance metadata
    for chunk in vs.all_chunks():
        assert {"source_file", "page", "document_id", "original_language", "translated"} <= set(chunk["metadata"])
    assert vs.ntotal == sum(r.chunks for r in reports.values())
    assert len(vs.list_documents()) == 4

    # persisted, and re-ingesting the same files is rejected as duplicate
    pipe2, vs2 = _pipeline(settings, fake_embedder)
    assert vs2.ntotal == vs.ntotal
    again = pipe2.ingest([corpus / "Article6_glossary.txt"])
    assert again[0].status == "rejected" and "duplicate" in again[0].errors[0]
    assert vs2.ntotal == vs.ntotal


def test_translation_failure_marks_file_failed(settings, fake_embedder, sample_dir):
    class Boom:
        def translate(self, text, lang):
            raise RuntimeError("model missing")

    pipe, vs = _pipeline(settings, fake_embedder, Boom())
    [r] = pipe.ingest([sample_dir / "Accord_de_Paris-extrait-FR.pdf"])
    assert r.status == "failed" and "model missing" in r.errors[0]
    assert vs.ntotal == 0
