"""Unpaywall provider — resolves open-access availability and full-text PDF URLs by DOI."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import config
from shared.models import Paper

logger = logging.getLogger(__name__)

BASE_URL = "https://api.unpaywall.org/v2"

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_TIMEOUT = 10  # seconds per request


def _clean_doi(doi: str | None) -> str | None:
    """Strip standard DOI URL prefixes and whitespace."""
    if not doi:
        return None
    cleaned = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return cleaned or None


def get_availability(doi: str) -> dict | None:
    """Fetch availability record for a single DOI from Unpaywall API.

    Returns parsed JSON dictionary on success, or None on failure/missing email.
    Never raises exceptions.
    """
    if not config.UNPAYWALL_EMAIL or not config.UNPAYWALL_EMAIL.strip():
        logger.warning("UNPAYWALL_EMAIL is not configured; skipping Unpaywall lookup")
        return None

    clean_doi = _clean_doi(doi)
    if not clean_doi:
        return None

    url = f"{BASE_URL}/{clean_doi}"
    params = {"email": config.UNPAYWALL_EMAIL.strip()}

    backoff = _INITIAL_BACKOFF
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("Unpaywall request failed for DOI %s: %s", clean_doi, exc)
            return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                logger.warning("Unpaywall returned invalid JSON for DOI %s", clean_doi)
                return None

        if resp.status_code == 429:
            logger.info(
                "Unpaywall rate limit (attempt %d/%d), backing off %.1fs",
                attempt + 1,
                _MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 404:
            logger.info("DOI not found in Unpaywall (404): %s", clean_doi)
            return None

        if resp.status_code == 422:
            logger.warning("Unpaywall unprocessable DOI (422): %s", clean_doi)
            return None

        logger.warning(
            "Unpaywall HTTP %d for DOI %s: %s",
            resp.status_code,
            clean_doi,
            resp.text[:200],
        )
        return None

    logger.warning("Unpaywall rate limit exceeded after %d retries for DOI %s", _MAX_RETRIES, clean_doi)
    return None


def enrich_paper(paper: Paper) -> Paper:
    """Enrich a single Paper with Open Access status and PDF URL from Unpaywall.

    If the paper has no DOI, if UNPAYWALL_EMAIL is missing, or if lookup fails,
    returns the paper with existing metadata preserved.
    """
    if not paper.doi:
        return paper

    data = get_availability(paper.doi)
    if not data or not isinstance(data, dict):
        return paper

    # Extract OA status
    is_oa = data.get("is_oa")
    if is_oa is True:
        is_open_access = True
        access_status = "open"
    elif is_oa is False:
        is_open_access = False
        access_status = "paywalled"
    else:
        is_open_access = paper.is_open_access
        access_status = paper.access_status

    # Extract best available open-access PDF URL
    pdf_url: str | None = None
    best_oa = data.get("best_oa_location")
    if isinstance(best_oa, dict):
        pdf_url = best_oa.get("url_for_pdf")

    # Fallback to oa_locations list if best_oa_location did not supply a PDF URL
    if not pdf_url and isinstance(data.get("oa_locations"), list):
        for loc in data["oa_locations"]:
            if isinstance(loc, dict) and loc.get("url_for_pdf"):
                pdf_url = loc["url_for_pdf"]
                break

    # If still no direct url_for_pdf, check if best_oa url points to a PDF
    if not pdf_url and isinstance(best_oa, dict):
        candidate_url = best_oa.get("url")
        if candidate_url and isinstance(candidate_url, str) and candidate_url.lower().endswith(".pdf"):
            pdf_url = candidate_url

    # Preserve existing pdf_url if Unpaywall has none
    final_pdf_url = pdf_url if (pdf_url and isinstance(pdf_url, str) and pdf_url.strip()) else paper.pdf_url

    # Track unpaywall source
    sources = list(paper.sources)
    if "unpaywall" not in sources:
        sources.append("unpaywall")

    updated_data = paper.model_dump()
    updated_data["is_open_access"] = is_open_access
    updated_data["access_status"] = access_status
    updated_data["pdf_url"] = final_pdf_url
    updated_data["sources"] = sources

    return Paper(**updated_data)


def enrich_papers(papers: list[Paper]) -> list[Paper]:
    """Enrich a list of Paper objects with Unpaywall open-access data.

    Preserves order of papers. Skips papers without DOIs.
    """
    if not papers:
        return []

    papers_with_doi = [p for p in papers if p.doi]
    if len(papers_with_doi) > 1:
        with ThreadPoolExecutor(max_workers=min(5, len(papers_with_doi))) as executor:
            return list(executor.map(enrich_paper, papers))

    return [enrich_paper(p) for p in papers]
