"""
pdf_generator.py
================
Generate a PDF of RAG sources. Used by Streamlit (app.py) for inline PDF.
Primary path: Markdown → HTML → WeasyPrint (best layout; needs system Pango/Cairo).
Fallback: fpdf2 (pure Python; works on Streamlit Cloud when WeasyPrint native libs fail).
Returns Base64 for inline display.
"""

import base64
import html as html_module
import pathlib
import re
from typing import List, Optional, Tuple

import markdown
from langchain_core.documents import Document

_FEEDBACK_PATTERNS = [
    r"Helpful\?\s*Yes\s*No.*",  # Apple feedback boilerplate
    r"Character limit:\s*\d+.*",
    r"Maximum character limit is\s*\d+\..*",
    r"Submit Thanks for your feedback\..*",
    r"Give feedback about this article.*",
    r"Choose a section to give feedback on.*",
    r"^Next\s*$",
    r"^\[\s*Skip to main content\s*\].*",
]


def _try_weasyprint_html():
    """WeasyPrint needs GObject/Pango/Cairo; Cloud may lack them despite packages.txt."""
    try:
        from weasyprint import HTML

        return HTML
    except (OSError, ImportError):
        return None


def _clean_for_pdf(text: str) -> str:
    """Light cleanup for PDF readability; does not affect retrieval/indexing."""
    if not text:
        return ""
    out = text
    for pat in _FEEDBACK_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"\[\s*\]\([^)]+\)", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _clean_title(title: str) -> str:
    if not title:
        return ""
    t = title
    t = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:140]


def _html_to_plain(html: str) -> str:
    """Strip HTML tags for fpdf2 path (no HTML renderer)."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.I)
    text = re.sub(r"</li>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html_module.unescape(text)


def _fpdf_dejavu_paths() -> Tuple[Optional[str], Optional[str]]:
    """Locate bundled DejaVu TTFs inside fpdf2."""
    try:
        import fpdf as fpdf_mod
    except ImportError:
        return None, None
    root = pathlib.Path(fpdf_mod.__file__).parent
    for sub in ("font", "fonts"):
        d = root / sub
        if not d.is_dir():
            continue
        reg = d / "DejaVuSans.ttf"
        bold = d / "DejaVuSans-Bold.ttf"
        if reg.is_file():
            return str(reg), str(bold) if bold.is_file() else str(reg)
    return None, None


def _create_sources_pdf_fpdf2(docs: List[Document]) -> str:
    """Pure-Python PDF when WeasyPrint is unavailable."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    reg, bold = _fpdf_dejavu_paths()
    if reg:
        pdf.add_font("DejaVu", "", reg)
        pdf.add_font("DejaVu", "B", bold or reg)
        body_font = "DejaVu"
    else:
        pdf.set_font("Helvetica", size=11)
        body_font = "Helvetica"

    def set_body(size: int = 11, style: str = "") -> None:
        if body_font == "DejaVu":
            pdf.set_font("DejaVu", style, size)
        else:
            pdf.set_font("Helvetica", style, size)

    if not docs:
        pdf.add_page()
        set_body(11, "")
        pdf.multi_cell(0, 8, "No sources available.")
        raw = pdf.output(dest="S")
        if isinstance(raw, str):
            raw = raw.encode("latin-1")
        return base64.b64encode(raw).decode("ascii")

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        raw_title = (meta.get("title") or "").strip()
        title = _clean_title(raw_title) or meta.get("source", "") or f"Source {i}"
        url = meta.get("source", "") or ""
        body = _clean_for_pdf(doc.page_content or "")

        section_md = f"""## Source {i}: {title}

{url}

{body}
"""
        html_body = markdown.markdown(
            section_md,
            extensions=["extra", "sane_lists"],
            output_format="html5",
        )
        plain = _html_to_plain(html_body)

        pdf.add_page()
        set_body(12, "B")
        pdf.multi_cell(0, 8, f"Source {i}: {title}")
        set_body(10, "")
        if url:
            pdf.set_text_color(37, 99, 235)
            pdf.multi_cell(0, 6, url)
            pdf.set_text_color(0, 0, 0)
        set_body(11, "")
        pdf.multi_cell(0, 5, plain)

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    return base64.b64encode(raw).decode("ascii")


def _create_sources_pdf_weasyprint(docs: List[Document], HTML) -> str:
    """Markdown → HTML → WeasyPrint."""
    if not docs:
        empty_html = "<html><body><p>No sources available.</p></body></html>"
        pdf_bytes = HTML(string=empty_html).write_pdf()
        return base64.b64encode(pdf_bytes).decode("ascii")

    sections: List[str] = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        raw_title = (meta.get("title") or "").strip()
        title = _clean_title(raw_title) or meta.get("source", "") or f"Source {i}"
        url = meta.get("source", "") or ""
        body = _clean_for_pdf(doc.page_content or "")

        section_md = f"""## Source {i}: {title}

{url}

{body}
"""
        sections.append(section_md)

    markdown_text = "\n\n---\n\n".join(sections)

    html_body = markdown.markdown(
        markdown_text,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )

    html = f"""
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    margin: 18mm 14mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    font-size: 13px;
    line-height: 1.5;
    color: #111827;
  }}
  h1, h2, h3 {{
    color: #111827;
    margin-top: 0.8em;
    margin-bottom: 0.4em;
  }}
  h2 {{
    font-size: 1.1rem;
    page-break-after: avoid;
  }}
  p {{
    margin: 0.2em 0 0.4em 0;
  }}
  a {{
    color: #2563eb;
    text-decoration: none;
    word-break: break-word;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    background: #f3f4f6;
    padding: 1px 4px;
    border-radius: 4px;
  }}
  pre code {{
    display: block;
    padding: 10px 12px;
    border-radius: 8px;
    overflow-wrap: anywhere;
  }}
  ul, ol {{
    margin: 0.2em 0 0.6em 1.2em;
  }}
  img {{
    max-width: 75%;
    height: auto;
    display: block;
    margin: 12px auto;
    page-break-inside: avoid;
  }}
  img[src*="icon"],
  img[width="20"],
  img[height="20"],
  img[src*="=h36"],
  img[src*="=h24"] {{
    width: 16px !important;
    max-width: none !important;
    height: 16px !important;
    display: inline !important;
    margin: 0 4px !important;
    vertical-align: text-bottom;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 0.8em 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    pdf_bytes = HTML(string=html).write_pdf()
    return base64.b64encode(pdf_bytes).decode("ascii")


def create_sources_pdf(docs: List[Document]) -> str:
    """
    Build a PDF from LangChain Document objects (title, source URL, content).
    Uses WeasyPrint when native libs are available; otherwise fpdf2.
    Returns the PDF as a Base64-encoded string (Streamlit inline viewer).
    """
    HTML = _try_weasyprint_html()
    if HTML is not None:
        return _create_sources_pdf_weasyprint(docs, HTML)
    return _create_sources_pdf_fpdf2(docs)
