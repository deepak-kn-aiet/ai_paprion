"""OpenAlex provider — searches the Works API and normalizes to Paper."""

import logging
import time

import requests

import config
from shared.models import Paper, make_paper_id

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"

# Retry config for rate limiting
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_TIMEOUT = 10  # seconds per request


def search(
    query: str,
    limit: int = 10,
    offset: int = 0,
    page: int | None = None,
) -> list[Paper]:
    """Search OpenAlex works matching *query*.

    Supports both offset-based pagination and explicit 1-indexed page.
    Returns a list of Paper objects (may be empty on error or no results).
    Never raises — logs errors and returns [].
    """
    per_page = max(1, min(limit, 200))  # OpenAlex maximum is 200 per page

    if page is not None and page >= 1:
        page_num = page
    else:
        page_num = (offset // per_page) + 1 if per_page > 0 else 1

    params: dict = {
        "search": query,
        "per_page": per_page,
        "page": page_num,
    }

    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY

    headers = _build_headers()

    data = _get_with_retry(f"{BASE_URL}/works", params, headers)
    if data is None:
        return []

    raw_works = data.get("results") or []
    papers: list[Paper] = []
    for raw in raw_works:
        paper = _normalize(raw)
        if paper is not None:
            papers.append(paper)
    return papers


# --- internal helpers ---


def _build_headers() -> dict[str, str]:
    """Build request headers, adding User-Agent with polite pool email or API key."""
    headers: dict[str, str] = {}
    if config.OPENALEX_EMAIL:
        headers["User-Agent"] = f"ai_paprion (mailto:{config.OPENALEX_EMAIL})"
    else:
        headers["User-Agent"] = "ai_paprion"

    if config.OPENALEX_API_KEY:
        headers["api-key"] = config.OPENALEX_API_KEY

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
            logger.warning("OpenAlex request failed: %s", exc)
            return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                logger.warning("OpenAlex returned invalid JSON")
                return None

        if resp.status_code == 429:
            logger.info(
                "OpenAlex rate limit (attempt %d/%d), backing off %.1fs",
                attempt + 1, _MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            backoff *= 2
            continue

        # Other HTTP errors — don't retry
        logger.warning(
            "OpenAlex HTTP %d: %s", resp.status_code, resp.text[:200]
        )
        return None

    logger.warning("OpenAlex rate limit exceeded after %d retries", _MAX_RETRIES)
    return None


def _clean_doi(doi: str | None) -> str | None:
    """Strip DOI URL prefixes and whitespace."""
    if not doi:
        return None
    cleaned = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return cleaned or None


def _clean_arxiv_id(arxiv_val: str | None) -> str | None:
    """Strip arXiv URL prefixes if present."""
    if not arxiv_val:
        return None
    cleaned = arxiv_val.strip()
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "arxiv:",
    ):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return cleaned or None


def _clean_openalex_id(openalex_val: str | None) -> str | None:
    """Extract ID portion from an OpenAlex URI or string (e.g. W12345)."""
    if not openalex_val:
        return None
    cleaned = openalex_val.strip()
    for prefix in ("https://openalex.org/", "http://openalex.org/"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return cleaned or None


def _reconstruct_abstract(raw: dict) -> str | None:
    """Reconstruct abstract from abstract_inverted_index or return string abstract."""
    # Direct abstract string if provided
    abstract_str = raw.get("abstract")
    if isinstance(abstract_str, str) and abstract_str.strip():
        return abstract_str.strip()

    inverted_index = raw.get("abstract_inverted_index")
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None

    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    word_positions.append((pos, word))

    if not word_positions:
        return None

    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def _extract_authors(raw: dict) -> list[str]:
    """Extract author display names from authorships structure."""
    authorships = raw.get("authorships")
    if isinstance(authorships, list):
        names: list[str] = []
        for item in authorships:
            if isinstance(item, dict):
                author_obj = item.get("author")
                if isinstance(author_obj, dict):
                    name = author_obj.get("display_name")
                    if name and isinstance(name, str) and name.strip():
                        names.append(name.strip())
        if names:
            return names

    # Fallback to direct authors list if present
    raw_authors = raw.get("authors")
    if isinstance(raw_authors, list):
        names = []
        for a in raw_authors:
            if isinstance(a, dict) and a.get("name"):
                names.append(str(a["name"]).strip())
            elif isinstance(a, str) and a.strip():
                names.append(a.strip())
        return names

    return []


def _extract_venue(raw: dict) -> str | None:
    """Extract publication venue from primary_location or host_venue."""
    primary_location = raw.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            display_name = source.get("display_name")
            if display_name and isinstance(display_name, str) and display_name.strip():
                return display_name.strip()

    host_venue = raw.get("host_venue")
    if isinstance(host_venue, dict):
        display_name = host_venue.get("display_name")
        if display_name and isinstance(display_name, str) and display_name.strip():
            return display_name.strip()

    venue_val = raw.get("venue")
    if isinstance(venue_val, str) and venue_val.strip():
        return venue_val.strip()

    return None


def _extract_pdf_url(raw: dict) -> str | None:
    """Extract direct PDF URL from primary_location or open_access."""
    primary_location = raw.get("primary_location")
    if isinstance(primary_location, dict):
        pdf_url = primary_location.get("pdf_url")
        if pdf_url and isinstance(pdf_url, str) and pdf_url.strip():
            return pdf_url.strip()

    open_access = raw.get("open_access")
    if isinstance(open_access, dict):
        oa_url = open_access.get("oa_url")
        if oa_url and isinstance(oa_url, str) and oa_url.strip():
            return oa_url.strip()

    direct_pdf = raw.get("pdf_url")
    if isinstance(direct_pdf, str) and direct_pdf.strip():
        return direct_pdf.strip()

    return None


def _normalize(raw: dict) -> Paper | None:
    """Convert a single OpenAlex work dict into a Paper, or None if unusable."""
    # Title resolution
    title = raw.get("title") or raw.get("display_name")
    if not title or not isinstance(title, str) or not title.strip():
        return None

    # Identifier resolution
    ids_dict = raw.get("ids") if isinstance(raw.get("ids"), dict) else {}

    raw_doi = raw.get("doi") or ids_dict.get("doi")
    doi = _clean_doi(raw_doi)

    raw_arxiv = ids_dict.get("arxiv") or raw.get("arxiv")
    arxiv_id = _clean_arxiv_id(raw_arxiv)

    raw_openalex = ids_dict.get("openalex") or raw.get("id")
    openalex_id = _clean_openalex_id(raw_openalex)

    try:
        paper_id = make_paper_id(
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
        )
    except ValueError:
        return None  # No usable identifier

    # Abstract
    abstract = _reconstruct_abstract(raw)

    # Publication Year
    year = raw.get("publication_year")
    if year is None:
        year = raw.get("year")
    if not isinstance(year, int):
        try:
            year = int(year) if year is not None else None
        except (ValueError, TypeError):
            year = None

    # Venue
    venue = _extract_venue(raw)

    # Citation count
    citation_count = raw.get("cited_by_count")
    if citation_count is None:
        citation_count = raw.get("citation_count")
    if not isinstance(citation_count, int):
        try:
            citation_count = int(citation_count) if citation_count is not None else None
        except (ValueError, TypeError):
            citation_count = None

    # Authors
    authors = _extract_authors(raw)

    # Open Access info
    open_access = raw.get("open_access") if isinstance(raw.get("open_access"), dict) else {}
    is_open_access = open_access.get("is_oa")
    if is_open_access is None and "is_oa" in raw:
        is_open_access = raw.get("is_oa")

    # URL resolution
    primary_location = raw.get("primary_location") if isinstance(raw.get("primary_location"), dict) else {}
    url = (
        raw.get("doi")
        or primary_location.get("landing_page_url")
        or raw.get("id")
    )
    if url and not isinstance(url, str):
        url = str(url)

    # PDF URL resolution
    pdf_url = _extract_pdf_url(raw)

    return Paper(
        paper_id=paper_id,
        title=title.strip(),
        abstract=abstract,
        year=year,
        venue=venue,
        url=url,
        pdf_url=pdf_url,
        citation_count=citation_count,
        influential_citation_count=None,  # OpenAlex does not provide this
        authors=authors,
        is_open_access=is_open_access,
        access_status="unknown",  # availability check fills this later
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        citation_source="openalex",
        sources=["openalex"],
    )
