"""Shared Paper model — used by all three members."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A single research paper with normalized metadata."""

    # Required
    paper_id: str
    title: str

    # Optional metadata
    abstract: str | None = None
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    year: int | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    review_score: float | None = None
    score: float | None = None  # filled by Member B only
    score_breakdown: dict[str, float] | None = None  # filled by Member B only
    is_recommended: bool | None = None  # filled by Member B only

    # Authors — safe mutable default via Field
    authors: list[str] = Field(default_factory=list)

    # Availability
    is_open_access: bool | None = None
    access_status: Literal["open", "paywalled", "unknown"] = "unknown"

    # Provider identifiers
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None

    # Which provider supplied the citation count
    citation_source: str | None = None

    # Which providers returned this paper
    sources: list[str] = Field(default_factory=list)


def make_paper_id(
    doi: str | None = None,
    arxiv_id: str | None = None,
    semantic_scholar_id: str | None = None,
    openalex_id: str | None = None,
) -> str:
    """Build a canonical paper_id from the best available identifier.

    Priority: DOI > arXiv ID > Semantic Scholar ID > OpenAlex ID.
    Raises ValueError if all are None.
    """
    if doi is not None:
        # Strip common URL prefixes, lowercase
        clean = doi.lower()
        for prefix in ("https://doi.org/", "http://doi.org/"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        return f"doi:{clean}"

    if arxiv_id is not None:
        return f"arxiv:{arxiv_id}"

    if semantic_scholar_id is not None:
        return f"s2:{semantic_scholar_id}"

    if openalex_id is not None:
        return f"openalex:{openalex_id}"

    raise ValueError("At least one identifier is required to create a paper_id")
