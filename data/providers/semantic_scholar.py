"""Semantic Scholar provider — searches the Graph API and normalizes to Paper."""

import logging
import time

import requests

import config
from shared.models import Paper, make_paper_id

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields we request from the API
_FIELDS = ",".join([
    "title", "abstract", "year", "venue", "authors",
    "citationCount", "influentialCitationCount",
    "externalIds", "openAccessPdf", "url", "isOpenAccess",
])

# Retry config for rate limiting
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_TIMEOUT = 10  # seconds per request


def search(query: str, limit: int = 10, offset: int = 0) -> list[Paper]:
    """Search Semantic Scholar for papers matching *query*.

    Returns a list of Paper objects (may be empty on error or no results).
    Never raises — logs errors and returns [].
    """
    params: dict = {
        "query": query,
        "fields": _FIELDS,
        "offset": offset,
        "limit": min(limit, 100),  # API max is 100 per page
    }
    headers = _build_headers()

    data = _get_with_retry(f"{BASE_URL}/paper/search", params, headers)
    if data is None:
        return []

    raw_papers = data.get("data") or []
    papers: list[Paper] = []
    for raw in raw_papers:
        paper = _normalize(raw)
        if paper is not None:
            papers.append(paper)
    return papers


# --- internal helpers ---


def _build_headers() -> dict[str, str]:
    """Build request headers, adding API key if available."""
    headers: dict[str, str] = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
    return headers


def _get_with_retry(
    url: str,
    params: dict,
    headers: dict[str, str],
) -> dict | None:
    """GET with retry on 429. Returns parsed JSON or None on failure."""
    backoff = _INITIAL_BACKOFF
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=_TIMEOUT
            )
        except requests.RequestException as exc:
            logger.warning("Semantic Scholar request failed: %s", exc)
            return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                logger.warning("Semantic Scholar returned invalid JSON")
                return None

        if resp.status_code == 429:
            logger.info(
                "Semantic Scholar rate limit (attempt %d/%d), backing off %.1fs",
                attempt + 1, _MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            backoff *= 2
            continue

        # Other HTTP errors — don't retry
        logger.warning(
            "Semantic Scholar HTTP %d: %s", resp.status_code, resp.text[:200]
        )
        return None

    logger.warning("Semantic Scholar rate limit exceeded after %d retries", _MAX_RETRIES)
    return None


def _normalize(raw: dict) -> Paper | None:
    """Convert a single S2 API result dict into a Paper, or None if unusable."""
    title = raw.get("title")
    if not title:
        return None  # title is required

    # Extract identifiers from externalIds
    ext = raw.get("externalIds") or {}
    doi = ext.get("DOI")
    arxiv_id = ext.get("ArXiv")
    s2_id = raw.get("paperId")

    # Build canonical paper_id
    try:
        paper_id = make_paper_id(
            doi=doi, arxiv_id=arxiv_id, semantic_scholar_id=s2_id
        )
    except ValueError:
        return None  # no usable identifier at all

    # Extract authors as list of name strings
    authors_raw = raw.get("authors") or []
    authors = [a["name"] for a in authors_raw if a.get("name")]

    # Open-access PDF
    oa_pdf = raw.get("openAccessPdf") or {}
    pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None

    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=raw.get("abstract"),
        year=raw.get("year"),
        venue=raw.get("venue") or None,  # API returns "" for missing
        url=raw.get("url"),
        pdf_url=pdf_url,
        citation_count=raw.get("citationCount"),
        influential_citation_count=raw.get("influentialCitationCount"),
        authors=authors,
        is_open_access=raw.get("isOpenAccess"),
        access_status="unknown",  # availability check fills this later
        doi=doi,
        arxiv_id=arxiv_id,
        semantic_scholar_id=s2_id,
        citation_source="semantic_scholar",
        sources=["semantic_scholar"],
    )
