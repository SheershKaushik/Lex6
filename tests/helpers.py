from pathlib import Path

import docx
import pymupdf


def make_pdf(path: Path, pages: list[str], password: str | None = None) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    if password:
        doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    else:
        doc.save(path)
    return path


def make_docx(path: Path, paragraphs: list[str]) -> Path:
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(path)
    return path
