"""
pdf_generator.py
================
Generate a PDF of RAG sources. Used by Streamlit (app.py) for inline PDF.
Simple path: Markdown → HTML → WeasyPrint. Returns Base64 for display.
"""

import base64
import re
from typing import List

import markdown
from weasyprint import HTML
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


def _clean_for_pdf(text: str) -> str:
    """Light cleanup for PDF readability; does not affect retrieval/indexing."""
    if not text:
        return ""
    out = text
    for pat in _FEEDBACK_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.MULTILINE)
    # Remove empty markdown links: [](...)
    out = re.sub(r"\[\s*\]\([^)]+\)", "", out)
    # Collapse excessive blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _clean_title(title: str) -> str:
    if not title:
        return ""
    t = title
    # Strip markdown images from titles so they don't wrap badly
    t = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:140]


def create_sources_pdf(docs: List[Document]) -> str:
    """
    Build a PDF from LangChain Document objects (title, source URL, content)
    using a Markdown → HTML → WeasyPrint pipeline.
    Returns the PDF as a Base64-encoded string (Streamlit inline viewer).
    """
    if not docs:
        # Return an empty but valid PDF document
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

    # Convert combined markdown to HTML
    html_body = markdown.markdown(
        markdown_text,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )

    # Wrap in a simple HTML template; let the browser-like renderer handle layout & images
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
  /* Google step icons commonly include '=h36' in the URL */
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
