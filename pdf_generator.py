"""
pdf_generator.py
================
Generate a PDF of RAG sources. Used by Streamlit (app.py) for inline PDF.
Returns Base64 for display; formatting is done locally (regex); no API calls.
# Evolution API (server.py) integration commented out — Streamlit-only.
"""
import base64
import io
import re
from typing import List

from fpdf import FPDF
from langchain_core.documents import Document


def _format_source_content(text: str) -> str:
    """
    Format raw RAG page_content for PDF: clean markdown images/links, normalize bullets.
    Handles multiline markdown and empty links. All local (regex); no API calls.
    """
    if not text or not text.strip():
        return text
    # Allow newlines/whitespace between ] and ( so wrapped markdown still matches
    _link_tail = r"\]\s*\(\s*[^)]+\)"
    # Replace markdown images ![alt](url) with readable label (alt can be empty)
    def _image_repl(m):
        alt = (m.group(1) or "").strip()
        return f"[Image: {alt}]" if alt else "[Image]"
    text = re.sub(r"!\[([^\]]*)" + _link_tail, _image_repl, text, flags=re.DOTALL)
    # Replace empty links [](url) so they don't show as raw
    text = re.sub(r"\[\s*\]\s*\(\s*[^)]+\)", " ", text, flags=re.DOTALL)
    # Replace markdown links [label](url) with just label (URL is in section header)
    text = re.sub(r"\[([^\]]+)" + _link_tail, r"\1", text, flags=re.DOTALL)
    # Normalize bullet lines: * or - at start of line -> hyphen
    text = re.sub(r"^[\*\-]\s+", "- ", text, flags=re.MULTILINE)
    # Collapse multiple spaces/newlines from removals
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def create_sources_pdf(docs: List[Document]) -> str:
    """
    Build a PDF from LangChain Document objects (title, source URL, content).
    Returns the PDF as a Base64-encoded string (Streamlit inline viewer).
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        if not title or len(title) < 2:
            title = meta.get("source", "") or f"Source {i}"
        source_url = meta.get("source", "") or ""

        # Section header: number and title
        pdf.set_font("Helvetica", "B", size=11)
        pdf.multi_cell(0, 6, f"Source {i}: {title}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        if source_url:
            pdf.set_text_color(0, 0, 180)
            pdf.multi_cell(0, 5, source_url, new_x="LMARGIN", new_y="NEXT", link=source_url)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Content: format for readability, then sanitize for Latin-1
        text = _format_source_content(doc.page_content or "")
        if text:
            try:
                text = text.encode("latin-1", errors="replace").decode("latin-1")
            except Exception:
                text = "".join(c if ord(c) < 256 else "?" for c in text)
            pdf.set_font("Helvetica", size=9)
            pdf.multi_cell(0, 5, text[:8000], new_x="LMARGIN", new_y="NEXT")  # cap per chunk

        pdf.ln(4)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
