"""
services.factcheck_service
===========================
Thin wrapper around the Google Fact Check Tools API.

Used by `tools.verify_claim_tool.verify_claim`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional, Dict

import requests

FACTCHECK_CLAIMS_SEARCH_URL = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)
FACTCHECK_CLAIMS_IMAGE_SEARCH_URL = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:imageSearch"
)


class FactCheckUnavailableError(RuntimeError):
    """Raised when the Fact Check service is unavailable or errors out."""



def _get_google_fact_api_key() -> str:
    """
    Prefer `GOOGLE_FACT_API_KEY`, but keep backward compatibility for this repo's
    existing `.env` key name (`Google_fact_api_key`).
    """

    key = os.environ.get("GOOGLE_FACT_API_KEY") or os.environ.get("Google_fact_api_key")
    return (key or "").strip()


def search_fact_check(query: str) -> Optional[Dict[str, Any]]:
    """
    Call Google Fact Check Tools `claims:search`.

    Returns the parsed JSON response dict on success.

    Returns None only when `query` is empty.

    Raises `FactCheckUnavailableError` when the API key is missing or the API
    call fails after retries.
    """

    query = (query or "").strip()
    if not query:
        return None

    api_key = _get_google_fact_api_key()
    if not api_key:
        raise FactCheckUnavailableError("Missing GOOGLE_FACT_API_KEY")

    # Retries: max 3 with exponential backoff.
    max_attempts = 3
    backoff_s = 1.0
    timeout_s = 30

    params = {"query": query, "key": api_key}

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                FACTCHECK_CLAIMS_SEARCH_URL, params=params, timeout=timeout_s
            )

            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception as e:
                    raise FactCheckUnavailableError("Invalid JSON from Fact Check") from e

            # Retry on rate-limit / transient server errors.
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < max_attempts:
                    time.sleep(backoff_s)
                    backoff_s *= 2
                    continue
            raise FactCheckUnavailableError(f"Fact Check API error: {resp.status_code}")
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue
            raise FactCheckUnavailableError("Fact Check API call failed") from e

    # Should be unreachable, but keep a safe fallback.
    _ = last_err
    raise FactCheckUnavailableError("Fact Check API call failed")


def search_fact_check_image(image_uri: str) -> Optional[Dict[str, Any]]:
    """
    Call Google Fact Check Tools `claims:imageSearch`.

    Returns the parsed JSON response dict on success, otherwise None.
    Raises FactCheckUnavailableError when API call fails after retries.
    """

    image_uri = (image_uri or "").strip()
    if not image_uri:
        return None

    api_key = _get_google_fact_api_key()
    if not api_key:
        raise FactCheckUnavailableError("Missing GOOGLE_FACT_API_KEY")

    max_attempts = 3
    backoff_s = 1.0
    timeout_s = 30

    params = {"imageUri": image_uri, "key": api_key}

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                FACTCHECK_CLAIMS_IMAGE_SEARCH_URL, params=params, timeout=timeout_s
            )

            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception as e:
                    raise FactCheckUnavailableError("Invalid JSON from Fact Check") from e

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < max_attempts:
                    time.sleep(backoff_s)
                    backoff_s *= 2
                    continue
            raise FactCheckUnavailableError(f"Fact Check API error: {resp.status_code}")
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue
            raise FactCheckUnavailableError("Fact Check API call failed") from e

    _ = last_err
    raise FactCheckUnavailableError("Fact Check API call failed")

