from langchain_core.documents import Document

from lex6.ingestion.chunking import split_documents
from lex6.ingestion.loaders import discover_files, extract_country_from_filename, load_file
from tests.helpers import make_docx, make_pdf


def test_load_pdf_docx_txt(tmp_path):
    pdf = load_file(make_pdf(tmp_path / "a.pdf", ["Page one text", "Page two text"]), "pdf")
    assert [d.metadata["page"] for d in pdf] == [0, 1]
    assert pdf[1].metadata["source_file"] == "a.pdf" and pdf[1].metadata["file_type"] == "pdf"
    assert "Page two" in pdf[1].page_content

    dx = load_file(make_docx(tmp_path / "b.docx", ["Para A", "Para B"]), "docx")
    assert len(dx) == 1 and "Para B" in dx[0].page_content and dx[0].metadata["page"] == 0

    t = tmp_path / "c.txt"
    t.write_text("plain text", encoding="utf-8")
    assert load_file(t, "text")[0].page_content == "plain text"


def test_discover_files(tmp_path):
    (tmp_path / "sub").mkdir()
    for n in ["a.pdf", "sub/b.DOCX", "c.txt", "d.xlsx"]:
        (tmp_path / n).write_text("x")
    assert [p.name for p in discover_files(tmp_path)] == ["a.pdf", "c.txt", "b.DOCX"]


def test_country_from_filename_word_boundaries():
    assert extract_country_from_filename("Romania_NDC.pdf") == "Romania"
    assert extract_country_from_filename("Niger-NDC-2021.docx") == "Niger"
    assert extract_country_from_filename("Nigeria_NDC.txt") == "Nigeria"
    assert extract_country_from_filename("Paris_agreement-English.pdf") is None


def test_split_documents_sizes_and_index():
    text = " ".join(f"word{i}" for i in range(1500))
    chunks = split_documents([Document(page_content=text, metadata={"source_file": "x", "page": 3})], 1000, 200)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1000 for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["page"] == 3 for c in chunks)
    # overlap: end of chunk 0 reappears at the start of chunk 1
    assert chunks[0].page_content[-50:].split()[-1] in chunks[1].page_content
