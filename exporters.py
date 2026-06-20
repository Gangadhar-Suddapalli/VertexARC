"""Export conversation history to TXT or PDF."""
from __future__ import annotations

from datetime import datetime

from fpdf import FPDF


def _header_lines(doc_name: str, summary: str | None) -> list[str]:
    lines = [
        "VertexArc PDF Q&A - Conversation Export",
        f"Document: {doc_name}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]
    if summary:
        lines += ["DOCUMENT SUMMARY", "-" * 60, summary, "", "=" * 60, ""]
    return lines


def to_text(history: list[dict], doc_name: str, summary: str | None = None) -> str:
    """Render the full conversation as plain text."""
    lines = _header_lines(doc_name, summary)
    if not history:
        lines.append("(No questions asked yet.)")
    for i, turn in enumerate(history, start=1):
        lines.append(f"Q{i}: {turn['question']}")
        lines.append(f"A{i}: {turn['answer']}")
        if turn.get("sources"):
            pages = ", ".join(str(p) for p in turn["sources"])
            lines.append(f"    (Source pages: {pages})")
        lines.append("")
    return "\n".join(lines)


def _safe(text: str) -> str:
    # FPDF core fonts are latin-1 only; replace unsupported chars gracefully.
    return text.encode("latin-1", "replace").decode("latin-1")


def to_pdf(history: list[dict], doc_name: str, summary: str | None = None) -> bytes:
    """Render the conversation as a PDF and return raw bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Effective text width; reset x to the left margin before every block so a
    # full-width multi_cell always has positive horizontal space to work with.
    eff_w = pdf.w - pdf.l_margin - pdf.r_margin

    def block(text: str, height: float = 6.0):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(eff_w, height, _safe(text))

    pdf.set_font("Helvetica", "B", 16)
    block("VertexArc PDF Q&A - Conversation", 10)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    block(f"Document: {doc_name}")
    block(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    if summary:
        pdf.set_font("Helvetica", "B", 12)
        block("Document Summary", 8)
        pdf.set_font("Helvetica", "", 11)
        block(summary)
        pdf.ln(4)

    if not history:
        pdf.set_font("Helvetica", "I", 11)
        block("(No questions asked yet.)")

    for i, turn in enumerate(history, start=1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 20, 20)
        block(f"Q{i}: {turn['question']}")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        block(turn["answer"])
        if turn.get("sources"):
            pages = ", ".join(str(p) for p in turn["sources"])
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            block(f"Source pages: {pages}", 5)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    out = pdf.output()  # fpdf2 returns a bytearray
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode("latin-1")
