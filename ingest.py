"""
ingest.py
=========
Modern (2025–2026) ingestion pipeline for Saksham MVP:
- Dynamic URL discovery via DuckDuckGo (optional --topic).
- LLM-ready Markdown via Jina Reader API (no Playwright/BeautifulSoup).
- Semantic chunking: MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter fallback.
- FAISS vector store with metadata: source, platform, header_path.

Storage: ./care_vector_db/ios18 | pixel | combined

Usage:
  python ingest.py                     # full ingest (hardcoded URL lists)
  python ingest.py --topic "ringtone"  # discover URLs for topic, ingest (append)
  python ingest.py --pixel-only
  python ingest.py --merge-only
  python ingest.py --topic "battery" --pixel-only
"""

import asyncio
import os
import shutil
import argparse
import time
from typing import List, Literal

from dotenv import load_dotenv
import requests

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DB = "./care_vector_db"
IOS_DB = os.path.join(BASE_DB, "ios18")
PIXEL_DB = os.path.join(BASE_DB, "pixel")
COMBINED_DB = os.path.join(BASE_DB, "combined")

# ─────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─────────────────────────────────────────
# CRAWL4AI: clean markdown with image src and link URLs kept (for PDF screenshots/icons)
# ─────────────────────────────────────────
_crawl4ai_md_generator = DefaultMarkdownGenerator(
    options={
        "ignore_links": False,   # keep [text](url) and link hrefs
        "ignore_images": False,  # keep ![alt](image_src) in markdown
        "escape_html": False,
        "body_width": 0,         # no wrapping
        "skip_internal_links": False,
    },
)
CRAWL4AI_CONFIG = CrawlerRunConfig(markdown_generator=_crawl4ai_md_generator)


def extract_markdown(result) -> str:
    """Get raw_markdown (or plain markdown string) from Crawl4AI result.

    We deliberately avoid fit_markdown to keep content as close as possible
    to the original page while still in markdown form.
    """
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    if hasattr(md, "raw_markdown") and md.raw_markdown:
        return md.raw_markdown or ""
    if isinstance(md, str):
        return md
    return ""


# ─────────────────────────────────────────
# HARDCODED URL LISTS (fallback when no --topic)
# ─────────────────────────────────────────
APPLE_URLS = [
    "https://support.apple.com/en-in/guide/iphone/iph4fd8a0b89/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph2968440de/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcfadf0701/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph62faab6a4/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph14c31d991/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9bfec93b1/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphca3d8b4e3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6d50ec543/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph8739025dd/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3c9951d7/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph16ecebf48/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6cfaf98b6/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph0e5ca7dd3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1ac0b35f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph14a867ae/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph315e0d58d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6a6decb13/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9e04f3be2/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9ccdd1bab/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph76d37ce7d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1e466ebee/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph07c867f28/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1ac0b4af/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3c8f64e92/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph5fe280d90/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph584ea27f5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph75f461ff0/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph35c335575/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph060d99f5e/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph0ede5b8d6/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph629d2cd37/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph70107aec2/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph22d98bbca/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphbd435673d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph37fdd714b/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4ce326e9d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9a847efc7/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7d116e557/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/ipha0706e2bc/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1d019df4e/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph60d026b44/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph59095ec58/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph8f357526d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph21a030ae3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3bf19d7b9/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphc61044c11/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe9b48b89e/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf9219d8c9/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphfaf30bdbd/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphea8b95631/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7f9e64962/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph2ab28320d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4cad323fe/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph5f869c0af/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7df5983ee/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7778f2888/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4fff7afb5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphfc2cd4c5f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf699c1ffc/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe9d46e90f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph21addc265/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph120bd1b3b/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphb9c2b948f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf02627316/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphfd5b616b5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9b7f53382/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3e2e23d1/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1fbef4daa/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3afc3b3fc/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph212965adb/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4df1c0dad/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6ef6ef377/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph220ea8dca/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph206c570e3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe7aa3336/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4bda412a5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6a8e168cb/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6013e96f4/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph0db74c881/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphaff1d606/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphc23ad3473/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphb3100d149/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph392d77d5f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcd5447866/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcd8d3c813/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf28f17237/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph29dbe3fb6/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph17dc31049/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7fe7a50a7/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph2e42d3117/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe7f70fba4/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9c11382ff/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph23f4d9aa9/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcce2770fd/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe22408524/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphc57feab64/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph02f94fc1c/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe0990f7bb/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphd5300a341/18.0/ios/18.0",
]

GOOGLE_URLS = [
    "https://support.google.com/pixelphone/answer/14140287",
    "https://support.google.com/pixelphone/answer/12967594",
    "https://support.google.com/pixelphone/answer/7158570",
    "https://support.google.com/pixelphone/answer/15199831",
    "https://support.google.com/pixelphone/answer/14116441",
    "https://support.google.com/pixelphone/answer/7535206",
    "https://support.google.com/pixelphone/answer/6111329",
    "https://support.google.com/pixelphone/answer/7444033",
    "https://support.google.com/pixelphone/answer/2819525",
    "https://support.google.com/pixelphone/answer/2818748",
    "https://support.google.com/pixelphone/answer/7680439",
    "https://support.google.com/pixelphone/answer/14782427",
    "https://support.google.com/pixelphone/answer/6183600",
    "https://support.google.com/pixelphone/answer/6187458",
    "https://support.google.com/pixelphone/answer/6006564",
    "https://support.google.com/pixelphone/answer/6122841",
    "https://support.google.com/pixelphone/answer/12913009",
    "https://support.google.com/pixelphone/answer/9316333",
    "https://support.google.com/pixelphone/answer/7283669",
    "https://support.google.com/pixelphone/answer/2844832",
    "https://support.google.com/pixelphone/answer/2781850",
    "https://support.google.com/pixelphone/answer/15182154",
    "https://support.google.com/pixelphone/answer/13202895",
    "https://support.google.com/pixelphone/answer/7055029",
    "https://support.google.com/pixelphone/answer/9118387",
    "https://support.google.com/pixelphone/answer/2819524",
    "https://support.google.com/pixelphone/answer/4596836",
    "https://support.google.com/pixelphone/answer/9218411",
    "https://support.google.com/pixelphone/answer/6187455",
    "https://support.google.com/pixelphone/answer/6090599",
    "https://support.google.com/pixelphone/answer/13675043",
    "https://support.google.com/pixelphone/answer/7106961",
]

# ─────────────────────────────────────────
# DYNAMIC URL DISCOVERY (DuckDuckGo)
# ─────────────────────────────────────────
def discover_support_urls(topic: str, platform: str) -> List[str]:
    """Discover up to 5 support URLs for a topic. platform: 'IOS18' or 'PIXEL'. Returns [] on network/rate-limit errors."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("  ⚠ duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []

    query_apple = f'site:support.apple.com/en-in/guide/iphone/ {topic} "ios 18"'
    query_pixel = f'site:support.google.com/pixelphone/ {topic}'
    query = query_apple if platform == "IOS18" else query_pixel
    urls: List[str] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                link = (r.get("href") or r.get("link") or "").strip()
                if link and link not in urls:
                    urls.append(link)
                    if len(urls) >= 5:
                        break
    except Exception as e:
        print(f"  ⚠ DuckDuckGo search failed (timeout/rate limit?): {e}")
        return []
    return urls[:5]


# ─────────────────────────────────────────
# FETCH VIA CRAWL4AI (clean markdown only; image/link URLs kept for PDF)
# ─────────────────────────────────────────
async def _fetch_one_url(url: str):
    """Fetch one URL with Crawl4AI. Returns markdown string or None on failure."""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url, config=CRAWL4AI_CONFIG)
        if not result or not getattr(result, "success", False):
            return None
        return extract_markdown(result)
    except Exception as e:
        print(f"      Crawl4AI error: {e}")
        return None


def fetch_markdown(url: str) -> str:
    """Fetch URL via Crawl4AI; return clean Markdown or empty string."""
    md = asyncio.run(_fetch_one_url(url))
    return md or ""


# ─────────────────────────────────────────
# SEMANTIC CHUNKING: Markdown headers + fallback splitter
# ─────────────────────────────────────────
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)
MAX_HEADER_CHUNK_SIZE = 1500


def _header_path(metadata: dict) -> str:
    """Build a single header_path string from splitter metadata."""
    parts = []
    for key in ["Header 1", "Header 2", "Header 3"]:
        v = metadata.get(key)
        if v:
            parts.append(v.strip())
    return " > ".join(parts) if parts else ""


def semantic_chunk(markdown: str, source_url: str, platform: str) -> List[Document]:
    """Split markdown by headers first; use RecursiveCharacterTextSplitter for oversized chunks. Attach source, platform, header_path, title. No HTML."""
    if not (markdown and markdown.strip()):
        return []

    try:
        header_docs = MARKDOWN_SPLITTER.split_text(markdown)
    except Exception:
        header_docs = [Document(page_content=markdown, metadata={})]

    chunks: List[Document] = []
    for doc in header_docs:
        content = doc.page_content.strip()
        if not content or len(content) < 50:
            continue
        meta = dict(doc.metadata)
        header_path_str = _header_path(meta)
        if len(content) <= MAX_HEADER_CHUNK_SIZE:
            meta["source"] = source_url
            meta["platform"] = platform
            meta["header_path"] = header_path_str
            first_line = next(
                (line.strip() for line in content.splitlines() if line.strip()),
                "untitled",
            )
            meta["title"] = first_line[:120]
            chunks.append(Document(page_content=content, metadata=meta))
        else:
            sub_docs = FALLBACK_SPLITTER.split_documents(
                [Document(page_content=content, metadata=meta)]
            )
            for sub in sub_docs:
                sub.metadata["source"] = source_url
                sub.metadata["platform"] = platform
                sub.metadata["header_path"] = header_path_str
                first_line = next(
                    (line.strip() for line in sub.page_content.splitlines() if line.strip()),
                    "untitled",
                )
                sub.metadata["title"] = first_line[:120]
                chunks.append(sub)
    return chunks


# ─────────────────────────────────────────
# GARBAGE FILTER
# ─────────────────────────────────────────
def _is_garbage_chunk(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return True
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    if not lines:
        return True
    version_keywords = {
        "ios 26", "ios 18", "ios 17", "select version:",
        "table of contents", "android 15", "android 14",
    }
    version_line_count = sum(
        1 for l in lines
        if l.lower() in version_keywords or l.lower().startswith("ios ")
    )
    if version_line_count / len(lines) > 0.4:
        return True
    link_line_count = sum(1 for l in lines if l.startswith("[") and "](http" in l)
    if link_line_count / len(lines) > 0.5:
        return True
    return False


# ─────────────────────────────────────────
# FETCH URLS VIA CRAWL4AI, CHUNK MARKDOWN ONLY (no HTML store)
# ─────────────────────────────────────────
async def _fetch_and_chunk_urls_async(urls: List[str], platform: str) -> List[Document]:
    all_chunks: List[Document] = []
    success_count = 0
    for url in urls:
        print(f"  🔎 Fetching: {url}")
        markdown = await _fetch_one_url(url)
        if not markdown or len(markdown) < 100:
            print("      ⚠ No content — skipping")
            continue
        chunks = semantic_chunk(markdown, url, platform)
        clean = [c for c in chunks if not _is_garbage_chunk(c.page_content)]
        discarded = len(chunks) - len(clean)
        all_chunks.extend(clean)
        success_count += 1
        print(f"      ✓ {len(clean)} chunks ({discarded} discarded)")
        await asyncio.sleep(0.3)
    print(f"\n  📄 Fetched {success_count}/{len(urls)} URLs → {len(all_chunks)} total chunks")
    return all_chunks


def _fetch_and_chunk_urls(urls: List[str], platform: str) -> List[Document]:
    return asyncio.run(_fetch_and_chunk_urls_async(urls, platform))


# ─────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────
def _save_report(platform: str, urls: List[str]) -> None:
    path = os.path.join(BASE_DB, f"{platform.lower()}_indexed_pages.txt")
    existing = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = [line.strip() for line in f if line.strip()]
    seen = set(existing)
    with open(path, "w", encoding="utf-8") as f:
        for u in existing:
            f.write(u + "\n")
        for u in urls:
            if u not in seen:
                seen.add(u)
                f.write(u + "\n")
    print(f"  📝 Report saved → {path}")


# ─────────────────────────────────────────
# CORE INGEST
# ─────────────────────────────────────────
def ingest_platform(
    urls: List[str],
    platform: str,
    save_path: str,
    clear_existing: bool = False,
) -> FAISS:
    print(f"\n{'='*60}")
    print(f"🚀  Ingesting  {platform}")
    print(f"{'='*60}")

    if clear_existing and os.path.exists(save_path):
        print(f"  🧹 Clearing {save_path}")
        shutil.rmtree(save_path)

    chunks = _fetch_and_chunk_urls(urls, platform)
    if not chunks:
        raise RuntimeError(f"No chunks produced for {platform}. Aborting.")

    print(f"\n  💾 Building FAISS index …")
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    os.makedirs(save_path, exist_ok=True)
    vs.save_local(save_path)
    print(f"  ✅ Saved FAISS index → {save_path}")
    _save_report(platform, urls)
    return vs


# ─────────────────────────────────────────
# APPEND CHUNKS TO EXISTING INDEX
# ─────────────────────────────────────────
def _append_to_index(chunks: List[Document], save_path: str, platform: str) -> None:
    if not chunks:
        return
    if not os.path.exists(save_path):
        vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    else:
        existing = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        new_vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
        existing.merge_from(new_vs)
        vs = existing
    os.makedirs(save_path, exist_ok=True)
    vs.save_local(save_path)
    print(f"  ✅ Saved/updated index → {save_path}")


# ─────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────
def merge_indexes() -> None:
    print(f"\n{'='*60}")
    print("🔀  Merging ios18 + pixel → combined")
    print(f"{'='*60}")
    if not os.path.exists(IOS_DB) or not os.path.exists(PIXEL_DB):
        raise FileNotFoundError("Both ios18 and pixel indexes must exist before merging.")
    print("  Loading ios18 …")
    ios_vs = FAISS.load_local(IOS_DB, embeddings, allow_dangerous_deserialization=True)
    print("  Loading pixel …")
    pixel_vs = FAISS.load_local(PIXEL_DB, embeddings, allow_dangerous_deserialization=True)
    print("  Merging …")
    ios_vs.merge_from(pixel_vs)
    if os.path.exists(COMBINED_DB):
        shutil.rmtree(COMBINED_DB)
    ios_vs.save_local(COMBINED_DB)
    print(f"  ✅ Combined index saved → {COMBINED_DB}")


# ─────────────────────────────────────────
# PUBLIC RETRIEVER LOADER
# ─────────────────────────────────────────
def load_retriever(
    platform: Literal["ios18", "pixel", "combined"] = "combined",
    k: int = 6,
    score_threshold: float = 0.30,
):
    path_map = {"ios18": IOS_DB, "pixel": PIXEL_DB, "combined": COMBINED_DB}
    path = path_map[platform]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index not found at '{path}'. Run ingest first.")
    vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Apple iOS 18 and/or Google Pixel docs into FAISS (Markdown + semantic chunking)."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Discover URLs for this topic and ingest (append) instead of using full URL lists.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pixel-only", action="store_true", help="Only ingest Pixel.")
    group.add_argument("--merge-only", action="store_true", help="Only rebuild combined index.")
    args = parser.parse_args()

    os.makedirs(BASE_DB, exist_ok=True)

    if args.merge_only:
        merge_indexes()
        print("\n🎉  Merge complete.")
        return

    if args.topic.strip():
        # Dynamic ingestion by topic
        topic = args.topic.strip()
        print(f"\n🔍 Dynamic ingestion for topic: «{topic}»")
        if args.pixel_only:
            urls_pixel = discover_support_urls(topic, "PIXEL")
            if urls_pixel:
                chunks = _fetch_and_chunk_urls(urls_pixel, "PIXEL")
                _append_to_index(chunks, PIXEL_DB, "PIXEL")
                _save_report("pixel", urls_pixel)
            merge_indexes()
        else:
            urls_apple = discover_support_urls(topic, "IOS18")
            urls_pixel = discover_support_urls(topic, "PIXEL")
            if urls_apple:
                chunks_ios = _fetch_and_chunk_urls(urls_apple, "IOS18")
                _append_to_index(chunks_ios, IOS_DB, "IOS18")
                _save_report("IOS18", urls_apple)
            if urls_pixel:
                chunks_pixel = _fetch_and_chunk_urls(urls_pixel, "PIXEL")
                _append_to_index(chunks_pixel, PIXEL_DB, "PIXEL")
                _save_report("pixel", urls_pixel)
            merge_indexes()
        print("\n🎉  Topic ingest complete.")
        return

    # Full ingest (hardcoded lists)
    if args.pixel_only:
        if not os.path.exists(IOS_DB):
            print("⚠️  No ios18 index found. Run without flags first.")
            return
        ingest_platform(GOOGLE_URLS, "PIXEL", PIXEL_DB, clear_existing=True)
        merge_indexes()
    else:
        ingest_platform(APPLE_URLS, "IOS18", IOS_DB, clear_existing=True)
        ingest_platform(GOOGLE_URLS, "PIXEL", PIXEL_DB, clear_existing=True)
        merge_indexes()

    print("\n🎉  Ingest complete.")
    print(f"    Indexes: {IOS_DB}  |  {PIXEL_DB}  |  {COMBINED_DB}")


if __name__ == "__main__":
    main()
