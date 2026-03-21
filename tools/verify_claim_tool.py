"""
tools.verify_claim_tool
========================
Single tool exposed to the LLM for scam / misinformation verification.

Behavior:
- Calls Google Fact Check Tools API via `services.factcheck_service`.
- If results exist: formats a short fact-check summary string.
- If no results or API error: returns a placeholder fallback string.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from services.factcheck_service import (
    search_fact_check,
    search_fact_check_image,
    FactCheckUnavailableError,
)


_NO_RESULTS_FALLBACK = "No verified fact-check results found."

_API_UNAVAILABLE_FALLBACK = "Fact-check service is temporarily unavailable."


def _tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", (s or "").lower())


def _looks_similar(query: str, claim: str) -> bool:
    """
    Basic string similarity to reduce irrelevant Fact Check matches.
    No embeddings or extra APIs.
    """

    q = (query or "").strip().lower()
    c = (claim or "").strip().lower()
    if not q or not c:
        return False

    seq_ratio = SequenceMatcher(None, q, c).ratio()
    q_tokens = set(_tokenize(q))
    c_tokens = set(_tokenize(c))
    if not q_tokens or not c_tokens:
        return False

    jaccard = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))

    # Accept if either is reasonably high.
    return (seq_ratio >= 0.35) or (jaccard >= 0.15)


def _extract_top_result(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    # `claims:search` returns top-level `claims[]`.
    claims = payload.get("claims") or []
    if claims:
        claim0 = claims[0] or {}
    else:
        # `claims:imageSearch` returns top-level `results[]`, each containing `claim`.
        results = payload.get("results") or []
        if not results:
            return None
        claim0 = (results[0] or {}).get("claim") or {}

    claim_text = (claim0.get("text") or "").strip()

    claim_reviews = claim0.get("claimReview") or []
    if not claim_reviews:
        return None

    review0 = claim_reviews[0] or {}
    textual_rating = (review0.get("textualRating") or "").strip()
    publisher = review0.get("publisher") or {}
    publisher_name = (publisher.get("name") or "").strip()
    url = (review0.get("url") or "").strip()
    review_date = (review0.get("reviewDate") or "").strip()
    title = (review0.get("title") or "").strip()

    # Require at least one meaningful piece to avoid returning empty strings.
    if not (claim_text or textual_rating or url or publisher_name):
        return None

    return {
        "claim": claim_text,
        "verdict": textual_rating,
        "source": url,
        "publisher": publisher_name,
        "review_date": review_date,
        "title": title,
    }


@tool
def verify_claim(query: str) -> str:
    """
    Verify a potentially suspicious claim using Google Fact Check Tools.

    Args:
        query: The claim text (or the suspicious message content) to search for.

    Returns:
        A formatted fact-check string, or a placeholder fallback string if no results.
    """

    query = query or ""
    if not query.strip():
        return _NO_RESULTS_FALLBACK

    # When the app provides a public URL for an uploaded image, use imageSearch.
    image_uri = None
    prefix = "IMAGE_URI:"
    if query.strip().upper().startswith(prefix):
        image_uri = query.split(":", 1)[1].strip()
    image_mode = image_uri is not None

    try:
        payload = (
            search_fact_check_image(image_uri)
            if image_uri
            else search_fact_check(query)
        )
    except FactCheckUnavailableError:
        return _API_UNAVAILABLE_FALLBACK

    if not payload:
        return _NO_RESULTS_FALLBACK

    top = _extract_top_result(payload)
    if not top:
        return _NO_RESULTS_FALLBACK

    # If Fact Check returned an irrelevant match, treat it as "no results".
    # Skip this filter for imageSearch because `query` is an image URL, not claim text.
    if not image_mode and not _looks_similar(query, top.get("claim", "")):
        return _NO_RESULTS_FALLBACK

    return (
        "🔍 Fact Check Result:\n\n"
        f"🧾 Claim: {top['claim']}\n"
        f"📊 Rating: {top['verdict']}\n\n"
        f"📰 Publisher: {top['publisher']}\n"
        f"📅 Reviewed On: {top.get('review_date') or 'N/A'}\n\n"
        f"🧠 Title: {top.get('title') or 'N/A'}\n\n"
        f"🔗 Source: {top['source']}\n\n"
        "Note: This result is sourced directly from third-party fact-checking organizations."
    )

