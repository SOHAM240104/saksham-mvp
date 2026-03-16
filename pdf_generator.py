"""
pdf_generator.py
================
Generate a PDF of RAG sources. Used by Streamlit (app.py) for inline PDF.
This implementation renders via Markdown → HTML → WeasyPrint.
Returns Base64 for display.
"""

import base64
from typing import List

import markdown
from weasyprint import HTML
from langchain_core.documents import Document


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
        title = raw_title or meta.get("source", "") or f"Source {i}"
        url = meta.get("source", "") or ""
        body = doc.page_content or ""

        section_md = f"""## Source {i}: {title}

{url}

{body}
"""
        sections.append(section_md)

    markdown_text = "\n\n---\n\n".join(sections)

    # Convert combined markdown to HTML
    html_body = markdown.markdown(markdown_text)

    # Wrap in a simple HTML template; let the browser-like renderer handle layout & images
    html = f"""
<html>
<head>
<meta charset="utf-8">
<style>
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
  }}
  p {{
    margin: 0.2em 0 0.4em 0;
  }}
  ul, ol {{
    margin: 0.2em 0 0.6em 1.2em;
  }}
  img {{
    max-width: 500px;
    height: auto;
    margin: 0.4em 0;
    display: block;
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
