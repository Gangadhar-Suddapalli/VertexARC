"""PDF text extraction and chunking utilities."""
from __future__ import annotations

from dataclasses import dataclass

from pypdf import PdfReader

import config


@dataclass
class Chunk:
    text: str
    page: int          # 1-based page number the chunk starts on
    index: int         # position of the chunk in the document


def extract_pages(file) -> list[str]:
    """Extract text per page from a PDF file-like object or path.

    Returns a list where element i is the text of page (i+1).
    """
    reader = PdfReader(file)
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def _clean(text: str) -> str:
    # Collapse excessive whitespace while keeping paragraph structure readable.
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def chunk_pages(
    pages: list[str],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split page text into overlapping character chunks, tracking page numbers."""
    chunks: list[Chunk] = []
    idx = 0
    for page_no, raw in enumerate(pages, start=1):
        text = _clean(raw)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_no, index=idx))
                idx += 1
            if end == len(text):
                break
            start = end - overlap  # step back to create overlap
    return chunks


def document_stats(pages: list[str]) -> dict:
    full = "\n".join(pages)
    words = len(full.split())
    return {
        "num_pages": len(pages),
        "num_words": words,
        "num_chars": len(full),
    }
