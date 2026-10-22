"""Unified provider aggregator and deduplication service for literature search."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from data import cache
from data.providers import openalex, semantic_scholar, unpaywall
from shared.models import Paper, make_paper_id

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("semantic_scholar", "openalex")


def normalize_doi(doi: str | None) -> str:
    """Normalize DOI by lowercasing and stripping prefixes."""
    if not doi:
        return ""
    clean = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()
            break
    return clean


def normalize_arxiv_id(arxiv_id: str | None) -> str:
    """Normalize arXiv identifier by stripping URL prefixes and version suffixes."""
    if not arxiv_id:
        return ""
    clean = arxiv_id.strip().lower()
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "arxiv:",
    ):
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()
            break
    # Strip version suffix (e.g. 1706.03762v2 -> 1706.03762)
    clean = re.sub(r"v\d+$", "", clean)
    return clean


def normalize_title(title: str | None) -> str:
    """Normalize title for fuzzy fallback matching.
    
    Removes HTML tags, punctuation, and whitespace, lowercasing all letters.
    """
    if not title:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", title.lower())
    # Keep only alphanumeric characters
    text = re.sub(r"[^a-z0-9]", "", text)
    # If too short, do not match solely on title
    if len(text) < 3:
        return ""
    return text


def merge_papers(base: Paper, incoming: Paper) -> Paper:
    """Merge two duplicate Paper records into a single consolidated Paper.

    Preserves existing useful metadata without overwriting with missing or empty values.
    Combines provider identifiers and re-calculates the canonical paper_id.
    """
    data = base.model_dump()

    # Title: keep base title unless empty/whitespace and incoming is non-empty
    if (not data.get("title") or not data["title"].strip()) and incoming.title:
        data["title"] = incoming.title.strip()

    # Abstract: prefer non-empty; if both present, keep the longer, more complete one
    base_abstract = data.get("abstract")
    incoming_abstract = incoming.abstract
    if not base_abstract and incoming_abstract:
        data["abstract"] = incoming_abstract
    elif base_abstract and incoming_abstract:
        if len(incoming_abstract.strip()) > len(base_abstract.strip()):
            data["abstract"] = incoming_abstract

    # Venue
    if not data.get("venue") and incoming.venue:
        data["venue"] = incoming.venue

    # URLs
    if not data.get("url") and incoming.url:
        data["url"] = incoming.url
    if not data.get("pdf_url") and incoming.pdf_url:
        data["pdf_url"] = incoming.pdf_url

    # Year
    if data.get("year") is None and incoming.year is not None:
        data["year"] = incoming.year

    # Citation counts: keep higher count and credit the provider that reported it
    base_citations = data.get("citation_count")
    incoming_citations = incoming.citation_count
    if base_citations is None and incoming_citations is not None:
        data["citation_count"] = incoming_citations
        data["citation_source"] = incoming.citation_source
    elif base_citations is not None and incoming_citations is not None:
        if incoming_citations > base_citations:
            data["citation_count"] = incoming_citations
            data["citation_source"] = incoming.citation_source

    # Influential citations (e.g. from Semantic Scholar)
    if data.get("influential_citation_count") is None and incoming.influential_citation_count is not None:
        data["influential_citation_count"] = incoming.influential_citation_count

    # Review score
    if data.get("review_score") is None and incoming.review_score is not None:
        data["review_score"] = incoming.review_score

    # Score (Member B ranking score)
    if data.get("score") is None and incoming.score is not None:
        data["score"] = incoming.score

    # Authors: take the more complete list if incoming has more author entries
    base_authors = data.get("authors") or []
    incoming_authors = incoming.authors or []
    if not base_authors and incoming_authors:
        data["authors"] = list(incoming_authors)
    elif len(incoming_authors) > len(base_authors):
        data["authors"] = list(incoming_authors)

    # Open Access flag
    if data.get("is_open_access") is None and incoming.is_open_access is not None:
        data["is_open_access"] = incoming.is_open_access
    elif incoming.is_open_access is True:
        data["is_open_access"] = True

    # Access status
    if data.get("access_status") == "unknown" and incoming.access_status != "unknown":
        data["access_status"] = incoming.access_status

    # Provider identifiers: merge all non-empty IDs
    doi = data.get("doi") or incoming.doi
    arxiv_id = data.get("arxiv_id") or incoming.arxiv_id
    semantic_scholar_id = data.get("semantic_scholar_id") or incoming.semantic_scholar_id
    openalex_id = data.get("openalex_id") or incoming.openalex_id

    data["doi"] = doi
    data["arxiv_id"] = arxiv_id
    data["semantic_scholar_id"] = semantic_scholar_id
    data["openalex_id"] = openalex_id

    # Re-derive canonical paper_id based on priority: DOI > arXiv > S2 > OpenAlex
    try:
        data["paper_id"] = make_paper_id(
            doi=doi,
            arxiv_id=arxiv_id,
            semantic_scholar_id=semantic_scholar_id,
            openalex_id=openalex_id,
        )
    except ValueError:
        # Fall back to base paper_id if make_paper_id fails
        pass

    # Merge sources preserving order without duplicates
    merged_sources = list(data.get("sources") or [])
    for src in incoming.sources:
        if src not in merged_sources:
            merged_sources.append(src)
    data["sources"] = merged_sources

    return Paper(**data)


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers using DOI first, arXiv ID second, and normalized title as fallback.

    Merges metadata from duplicate records without replacing useful fields.
    Preserves the order of first appearance.
    """
    unique_papers: list[Paper] = []
    doi_map: dict[str, int] = {}
    arxiv_map: dict[str, int] = {}
    title_map: dict[str, int] = {}

    for paper in papers:
        matched_idx: int | None = None

        norm_doi = normalize_doi(paper.doi)
        norm_arxiv = normalize_arxiv_id(paper.arxiv_id)
        norm_title = normalize_title(paper.title)

        # 1. Match by DOI first
        if norm_doi and norm_doi in doi_map:
            matched_idx = doi_map[norm_doi]

        # 2. Match by arXiv ID second
        if matched_idx is None and norm_arxiv and norm_arxiv in arxiv_map:
            candidate_idx = arxiv_map[norm_arxiv]
            existing = unique_papers[candidate_idx]
            existing_doi = normalize_doi(existing.doi)
            # Ensure no conflicting DOIs
            if not (norm_doi and existing_doi and norm_doi != existing_doi):
                matched_idx = candidate_idx

        # 3. Match by normalized title as fallback
        if matched_idx is None and norm_title and norm_title in title_map:
            candidate_idx = title_map[norm_title]
            existing = unique_papers[candidate_idx]
            existing_doi = normalize_doi(existing.doi)
            existing_arxiv = normalize_arxiv_id(existing.arxiv_id)

            # Avoid false positive merges if DOIs or arXiv IDs explicitly conflict
            conflict = False
            if norm_doi and existing_doi and norm_doi != existing_doi:
                conflict = True
            if norm_arxiv and existing_arxiv and norm_arxiv != existing_arxiv:
                conflict = True

            if not conflict:
                matched_idx = candidate_idx

        # If duplicate found, merge into existing record
        if matched_idx is not None:
            merged = merge_papers(unique_papers[matched_idx], paper)
            unique_papers[matched_idx] = merged

            # Update index lookups with newly merged identifiers
            m_doi = normalize_doi(merged.doi)
            m_arxiv = normalize_arxiv_id(merged.arxiv_id)
            m_title = normalize_title(merged.title)
            if m_doi:
                doi_map[m_doi] = matched_idx
            if m_arxiv:
                arxiv_map[m_arxiv] = matched_idx
            if m_title:
                title_map[m_title] = matched_idx
        else:
            # Register new unique paper
            idx = len(unique_papers)
            unique_papers.append(paper)
            if norm_doi:
                doi_map[norm_doi] = idx
            if norm_arxiv:
                arxiv_map[norm_arxiv] = idx
            if norm_title:
                title_map[norm_title] = idx

    return unique_papers


def search(
    query: str,
    limit: int = 10,
    offset: int = 0,
    providers: list[str] | None = None,
    enrich_access: bool = False,
    use_cache: bool = True,
) -> list[Paper]:
    """Execute unified search across research paper providers and deduplicate results.

    Checks local SQLite cache first (2-day TTL).
    Invokes Semantic Scholar and OpenAlex in parallel on cache miss.
    Gracefully handles individual provider failures.
    Optionally enriches returned papers with Unpaywall open-access availability.
    Returns deduplicated, merged Paper objects up to *limit*.
    Never raises exceptions — logs errors and returns empty list on total failure.
    """
    clean_query = query.strip() if query else ""
    if not clean_query or limit <= 0:
        return []

    if providers is None:
        target_providers = list(SUPPORTED_PROVIDERS)
    else:
        target_providers = [p for p in providers if p in SUPPORTED_PROVIDERS]

    if not target_providers:
        logger.warning("No supported providers specified for query: %s", clean_query)
        return []

    # Check cache first
    cache_key = None
    if use_cache:
        try:
            cache_key = cache.make_query_key(
                clean_query,
                limit=limit,
                offset=offset,
                providers=target_providers,
                enrich_access=enrich_access,
            )
            cached_results = cache.get_cached_search(cache_key)
            if cached_results is not None:
                return cached_results
        except Exception as exc:
            logger.warning("Cache lookup error for query '%s': %s", clean_query, exc)

    combined_results: list[Paper] = []
    futures = {}
    successful_providers = 0

    with ThreadPoolExecutor(max_workers=len(target_providers)) as executor:
        for provider_name in target_providers:
            if provider_name == "semantic_scholar":
                futures[provider_name] = executor.submit(
                    semantic_scholar.search, clean_query, limit=limit, offset=offset
                )
            elif provider_name == "openalex":
                futures[provider_name] = executor.submit(
                    openalex.search, clean_query, limit=limit, offset=offset
                )

        for provider_name, future in futures.items():
            try:
                papers = future.result()
                successful_providers += 1
                if papers:
                    combined_results.extend(papers)
            except Exception as exc:
                logger.warning(
                    "Unified search provider '%s' failed unexpectedly: %s",
                    provider_name,
                    exc,
                )

    # Deduplicate and consolidate across all provider results
    deduped = deduplicate_papers(combined_results)
    limited = deduped[:limit]

    # Optional availability enrichment via Unpaywall
    final_results = limited
    if enrich_access and limited:
        try:
            final_results = unpaywall.enrich_papers(limited)
        except Exception as exc:
            logger.warning("Unpaywall availability enrichment failed unexpectedly: %s", exc)
            final_results = limited

    # Cache successful results (do not cache if all providers failed)
    if use_cache and cache_key and successful_providers > 0:
        try:
            cache.set_cached_search(cache_key, clean_query, final_results)
        except Exception as exc:
            logger.warning("Cache save error for query '%s': %s", clean_query, exc)

    return final_results


# Alias for convenience
unified_search = search
