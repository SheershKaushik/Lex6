from langchain_core.documents import Document

from lex6.ingestion.validation import FileValidator, validate_content
from tests.helpers import make_docx, make_pdf


def test_valid_files_pass(tmp_path):
    v = FileValidator()
    pdf = make_pdf(tmp_path / "a.pdf", ["Article 1. Hello world"])
    dx = make_docx(tmp_path / "b.docx", ["Article 2"])
    txt = tmp_path / "c.txt"
    txt.write_text("Article 3 text", encoding="utf-8")
    for path, kind in [(pdf, "pdf"), (dx, "docx"), (txt, "text")]:
        r = v.validate(path)
        assert r.ok, r.errors
        assert r.file_type == kind
        assert len(r.sha256) == 64


def test_missing_and_unsupported(tmp_path):
    v = FileValidator()
    assert "does not exist" in v.validate(tmp_path / "nope.pdf").errors[0]
    x = tmp_path / "sheet.xlsx"
    x.write_bytes(b"data")
    assert "unsupported file type" in v.validate(x).errors[0]


def test_empty_and_too_large(tmp_path):
    e = tmp_path / "e.txt"
    e.write_bytes(b"")
    assert FileValidator().validate(e).errors == ["file is empty"]
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * 2048)
    assert "limit" in FileValidator(max_file_size_mb=0.001).validate(big).errors[0]


def test_fake_pdf_and_docx_rejected(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_text("I am text pretending to be a PDF")
    assert "missing %PDF header" in FileValidator().validate(p).errors[0]
    d = tmp_path / "fake.docx"
    d.write_text("not a zip")
    assert "not a zip" in FileValidator().validate(d).errors[0]


def test_encrypted_pdf_rejected(tmp_path):
    p = make_pdf(tmp_path / "locked.pdf", ["secret"], password="pw")
    assert FileValidator().validate(p).errors == ["PDF is password protected"]


def test_binary_text_rejected(tmp_path):
    t = tmp_path / "bin.txt"
    t.write_bytes(b"abc\x00\x01\x02")
    assert "binary" in FileValidator().validate(t).errors[0]


def test_non_utf8_text_warns(tmp_path):
    t = tmp_path / "latin.txt"
    t.write_bytes("Accord de Paris – été".encode("cp1252"))
    r = FileValidator().validate(t)
    assert r.ok and r.warnings == ["text decoded as cp1252"]


def test_duplicate_rejected(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text("same content")
    first = FileValidator().validate(t)
    second = FileValidator(known_hashes={first.sha256}).validate(t)
    assert second.errors == ["duplicate: identical file already ingested"]


def test_validate_content():
    assert validate_content([]) == ["no content could be loaded"]
    assert "extractable text" in validate_content([Document(page_content="  ")])[0]
    assert validate_content([Document(page_content="x" * 50)]) == []
