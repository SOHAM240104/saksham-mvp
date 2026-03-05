"""
ingest_faiss.py
===============
Ingests Apple iOS 18 and Google Pixel documentation into FAISS vector stores.
Uses Playwright (headless Chromium) + BeautifulSoup to extract only the article
body from each page — no nav, no version selectors, no TOC, no chrome.

Storage layout (unchanged — fully compatible with existing app.py)
------------------------------------------------------------------
./care_vector_db/
    ios18/      ← FAISS index for Apple only
    pixel/      ← FAISS index for Pixel only
    combined/   ← merged index (both platforms)

Install dependencies (one-time)
--------------------------------
    pip install playwright beautifulsoup4 langchain-community langchain-openai faiss-cpu
    playwright install chromium

Usage
-----
    python ingest_faiss.py                  # full ingest
    python ingest_faiss.py --pixel-only     # re-ingest Pixel only
    python ingest_faiss.py --merge-only     # rebuild combined from existing indexes
"""

import os
import shutil
import argparse
import time
from typing import List, Literal

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ─────────────────────────────────────────
# PATHS  (unchanged from original)
# ─────────────────────────────────────────
BASE_DB     = "./care_vector_db"
IOS_DB      = os.path.join(BASE_DB, "ios18")
PIXEL_DB    = os.path.join(BASE_DB, "pixel")
COMBINED_DB = os.path.join(BASE_DB, "combined")

# ─────────────────────────────────────────
# EMBEDDINGS  (unchanged)
# ─────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─────────────────────────────────────────
# APPLE iOS 18 URLS  (unchanged)
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
    # Appended (also in APPLE_URLS_APPEND for --append-ios)
    "https://support.apple.com/en-in/guide/iphone/iph841379c3d/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iphb71f9b54d/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iphf574afb44/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph81c7fd7d1/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iphca3d8b4e3/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph1a1f981ad/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph83bfec492/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph9374b7411/ios",
    "https://support.apple.com/en-in/guide/iphone/iph37c04838/ios",
    "https://support.apple.com/en-in/guide/iphone/iph3d267104/ios",
]

# Additional Apple URLs to append to existing index (scrape only these when using --append-ios)
APPLE_URLS_APPEND = [
    "https://support.apple.com/en-in/guide/iphone/iph841379c3d/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iphb71f9b54d/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iphf574afb44/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph81c7fd7d1/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iphca3d8b4e3/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph1a1f981ad/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph83bfec492/26/ios/18",
    "https://support.apple.com/en-in/guide/iphone/iph9374b7411/ios",
    "https://support.apple.com/en-in/guide/iphone/iph37c04838/ios",
    "https://support.apple.com/en-in/guide/iphone/iph3d267104/ios",
]

# ─────────────────────────────────────────
# GOOGLE PIXEL URLS
# ─────────────────────────────────────────
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
    # Appended (also in GOOGLE_URLS_APPEND for --append-pixel)
    "https://support.google.com/pixelphone/answer/2819519?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/answer/2819577?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/answer/7289143?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/answer/7109524?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/?hl=en#topic=7083814",
]

# Additional Pixel URLs to append to existing index (scrape only these when using --append-pixel)
GOOGLE_URLS_APPEND = [
    "https://support.google.com/pixelphone/answer/2819519?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/answer/2819577?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/answer/7289143?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/answer/7109524?hl=en&ref_topic=7083814",
    "https://support.google.com/pixelphone/?hl=en#topic=7083814",
]

# ─────────────────────────────────────────
# TEXT SPLITTER  (unchanged)
# ─────────────────────────────────────────
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)


# ─────────────────────────────────────────
# APPLE PAGE EXTRACTOR
# Waits for the article body to render, then extracts only that content.
# Apple's article body is inside <section data-type="article"> after JS runs.
# Falls back through multiple selectors in case the structure varies.
# ─────────────────────────────────────────
def _extract_apple(page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=30000)

    # Wait for article content to appear — this is what FireCrawl was missing
    try:
        page.wait_for_selector("section[data-type='article'], #article, .article-body, main article", timeout=10000)
    except PlaywrightTimeout:
        pass  # Try to extract whatever rendered

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    # Try selectors in priority order — stop at first match
    selectors = [
        {"name": "section", "attrs": {"data-type": "article"}},
        {"name": "div",     "attrs": {"id": "article"}},
        {"name": "div",     "attrs": {"class": "article-body"}},
        {"name": "main",    "attrs": {}},
    ]

    for sel in selectors:
        el = soup.find(sel["name"], sel["attrs"] if sel["attrs"] else True)
        if el:
            # Remove nav, TOC, version selector, and feedback elements that
            # may be nested inside the article container
            for tag in el.find_all(["nav", "aside", "footer", "select",
                                     "script", "style", "noscript"]):
                tag.decompose()
            for tag in el.find_all(class_=["version-selector", "toc",
                                            "localnav", "breadcrumb",
                                            "feedback", "article-feedback"]):
                tag.decompose()

            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:  # Sanity check — real content is always longer
                return text

    # Last resort: body text (will have some chrome but better than nothing)
    body = soup.find("body")
    return body.get_text(separator="\n", strip=True) if body else ""


# ─────────────────────────────────────────
# GOOGLE PAGE EXTRACTOR
# Google support pages render content inside .article-body or [jscontroller]
# article divs. We wait for the main content container then extract it.
# ─────────────────────────────────────────
def _extract_google(page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=30000)

    try:
        page.wait_for_selector(".article-body, .cc-content, [jsname='WbKHeb'], #article-content", timeout=10000)
    except PlaywrightTimeout:
        pass

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    selectors = [
        {"name": "div", "attrs": {"class": "article-body"}},
        {"name": "div", "attrs": {"class": "cc-content"}},
        {"name": "div", "attrs": {"id":    "article-content"}},
        {"name": "main","attrs": {}},
    ]

    for sel in selectors:
        el = soup.find(sel["name"], sel["attrs"] if sel["attrs"] else True)
        if el:
            for tag in el.find_all(["nav", "aside", "footer", "select",
                                     "script", "style", "noscript"]):
                tag.decompose()
            for tag in el.find_all(class_=["related-articles", "feedback",
                                            "breadcrumb", "nav-list"]):
                tag.decompose()

            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text

    body = soup.find("body")
    return body.get_text(separator="\n", strip=True) if body else ""


# ─────────────────────────────────────────
# GARBAGE FILTER  (safety net after extraction)
# ─────────────────────────────────────────
def _is_garbage_chunk(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return True

    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    if not lines:
        return True

    version_keywords = {
        "ios 26", "ios 18", "ios 17", "ios 16", "ios 15",
        "ios 14", "ios 13", "ios 12", "select version:",
        "modifying this control", "table of contents",
        "android 15", "android 14", "android 13",
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
# METADATA ENRICHMENT
# ─────────────────────────────────────────
def _enrich_metadata(docs: List[Document], platform: str, source_url: str) -> List[Document]:
    for i, doc in enumerate(docs):
        first_line = next(
            (line.strip() for line in doc.page_content.splitlines() if line.strip()),
            "untitled"
        )
        doc.metadata.update({
            "platform":    platform,
            "source":      source_url,
            "chunk_index": i,
            "title":       first_line[:120],
        })
    return docs


# ─────────────────────────────────────────
# MAIN SCRAPER
# Opens one Playwright browser for all URLs in a batch.
# Uses the correct extractor per platform.
# ─────────────────────────────────────────
def _scrape_urls(urls: List[str], platform: str) -> List[Document]:
    all_chunks: List[Document] = []
    success_count = 0
    is_apple = platform == "IOS18"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()

        for url in urls:
            print(f"  🔎 Scraping: {url}")
            try:
                # Extract article body only
                if is_apple:
                    text = _extract_apple(page, url)
                else:
                    text = _extract_google(page, url)

                if not text or len(text.strip()) < 100:
                    print(f"      ⚠  No content extracted — skipping")
                    continue

                # Wrap in a Document then split
                raw_doc = Document(page_content=text, metadata={})
                enriched = _enrich_metadata([raw_doc], platform, url)
                chunks   = SPLITTER.split_documents(enriched)

                # Re-index and filter
                clean_chunks = []
                for j, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = j
                    if not _is_garbage_chunk(chunk.page_content):
                        clean_chunks.append(chunk)

                discarded = len(chunks) - len(clean_chunks)
                all_chunks.extend(clean_chunks)
                success_count += 1
                print(f"      ✓ {len(clean_chunks)} chunks ({discarded} discarded)")

                # Small polite delay between requests
                time.sleep(0.5)

            except Exception as exc:
                print(f"      ✗ Failed: {exc}")

        browser.close()

    print(f"\n  📄 Scraped {success_count}/{len(urls)} URLs → {len(all_chunks)} total chunks")
    return all_chunks


def _save_report(platform: str, urls: List[str]) -> None:
    path = os.path.join(BASE_DB, f"{platform.lower()}_indexed_pages.txt")
    with open(path, "w") as fh:
        for url in urls:
            fh.write(url + "\n")
    print(f"  📝 Report saved → {path}")


# ─────────────────────────────────────────
# CORE INGEST  (unchanged interface)
# ─────────────────────────────────────────
def ingest_platform(
    urls:           List[str],
    platform:       str,
    save_path:      str,
    clear_existing: bool = False,
) -> FAISS:
    print(f"\n{'='*60}")
    print(f"🚀  Ingesting  {platform}")
    print(f"{'='*60}")

    if clear_existing and os.path.exists(save_path):
        print(f"  🧹 Clearing {save_path}")
        shutil.rmtree(save_path)

    chunks = _scrape_urls(urls, platform)
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
# APPEND APPLE URLS  (scrape only new URLs, merge into existing ios18)
# ─────────────────────────────────────────
def _append_apple_urls() -> None:
    if not os.path.exists(IOS_DB):
        raise FileNotFoundError(
            f"ios18 index not found at {IOS_DB}. Run full ingest first (no flags)."
        )

    print(f"\n{'='*60}")
    print("📎  Appending Apple URLs to existing ios18 index")
    print(f"{'='*60}")
    print(f"  URLs to scrape: {len(APPLE_URLS_APPEND)}")

    chunks = _scrape_urls(APPLE_URLS_APPEND, "IOS18")
    if not chunks:
        print("  ⚠ No chunks produced; index unchanged.")
        return

    print(f"\n  📂 Loading existing ios18 index …")
    existing_vs = FAISS.load_local(IOS_DB, embeddings, allow_dangerous_deserialization=True)
    print(f"  🔨 Building small index from new chunks …")
    new_vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    print(f"  🔀 Merging new docs into ios18 …")
    existing_vs.merge_from(new_vs)
    existing_vs.save_local(IOS_DB)
    print(f"  ✅ Saved updated ios18 index → {IOS_DB}")

    # Append new URLs to indexed_pages report
    report_path = os.path.join(BASE_DB, "ios18_indexed_pages.txt")
    existing = []
    if os.path.exists(report_path):
        with open(report_path) as f:
            existing = [line.strip() for line in f if line.strip()]
    with open(report_path, "w") as f:
        for u in existing:
            f.write(u + "\n")
        for u in APPLE_URLS_APPEND:
            f.write(u + "\n")
    print(f"  📝 Report updated → {report_path}")

    print("\n  🔀 Rebuilding combined index …")
    merge_indexes()
    print("\n🎉  Append complete.")


# ─────────────────────────────────────────
# APPEND PIXEL URLS  (scrape only new URLs, merge into existing pixel)
# ─────────────────────────────────────────
def _append_pixel_urls() -> None:
    if not os.path.exists(PIXEL_DB):
        raise FileNotFoundError(
            f"pixel index not found at {PIXEL_DB}. Run full ingest first (no flags or --pixel-only)."
        )

    print(f"\n{'='*60}")
    print("📎  Appending Pixel URLs to existing pixel index")
    print(f"{'='*60}")
    print(f"  URLs to scrape: {len(GOOGLE_URLS_APPEND)}")

    chunks = _scrape_urls(GOOGLE_URLS_APPEND, "PIXEL")
    if not chunks:
        print("  ⚠ No chunks produced; index unchanged.")
        return

    print(f"\n  📂 Loading existing pixel index …")
    existing_vs = FAISS.load_local(PIXEL_DB, embeddings, allow_dangerous_deserialization=True)
    print(f"  🔨 Building small index from new chunks …")
    new_vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    print(f"  🔀 Merging new docs into pixel …")
    existing_vs.merge_from(new_vs)
    existing_vs.save_local(PIXEL_DB)
    print(f"  ✅ Saved updated pixel index → {PIXEL_DB}")

    # Append new URLs to indexed_pages report
    report_path = os.path.join(BASE_DB, "pixel_indexed_pages.txt")
    existing = []
    if os.path.exists(report_path):
        with open(report_path) as f:
            existing = [line.strip() for line in f if line.strip()]
    with open(report_path, "w") as f:
        for u in existing:
            f.write(u + "\n")
        for u in GOOGLE_URLS_APPEND:
            f.write(u + "\n")
    print(f"  📝 Report updated → {report_path}")

    print("\n  🔀 Rebuilding combined index …")
    merge_indexes()
    print("\n🎉  Pixel append complete.")


# ─────────────────────────────────────────
# MERGE  (unchanged)
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
# PUBLIC RETRIEVER LOADER  (unchanged)
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
# CLI  (unchanged)
# ─────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Apple iOS 18 and/or Google Pixel docs into FAISS."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pixel-only",  action="store_true",
                       help="Only re-ingest Pixel. Apple index must already exist.")
    group.add_argument("--merge-only",  action="store_true",
                       help="Skip scraping; just rebuild the combined index.")
    group.add_argument("--append-ios",  action="store_true",
                       help="Append APPLE_URLS_APPEND to existing ios18 index (no full re-scrape).")
    group.add_argument("--append-pixel",  action="store_true",
                       help="Append GOOGLE_URLS_APPEND to existing pixel index (no full re-scrape).")
    args = parser.parse_args()

    os.makedirs(BASE_DB, exist_ok=True)

    if args.merge_only:
        merge_indexes()
    elif args.append_ios:
        _append_apple_urls()
    elif args.append_pixel:
        _append_pixel_urls()
    elif args.pixel_only:
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