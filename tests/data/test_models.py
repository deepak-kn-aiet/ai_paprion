"""Tests for the shared Paper model and make_paper_id helper."""

import pytest
from pydantic import ValidationError

from shared.models import Paper, make_paper_id

# --- Paper creation ---

def test_minimal_paper_defaults():
    """Only paper_id and title required; all optional fields are None or defaults."""
    p = Paper(paper_id="doi:10.1234/test", title="Test Paper")
    assert p.paper_id == "doi:10.1234/test"
    assert p.title == "Test Paper"
    # Numeric fields are None, not 0
    assert p.citation_count is None
    assert p.influential_citation_count is None
    assert p.year is None
    assert p.review_score is None
    assert p.score is None
    # String fields are None
    assert p.abstract is None
    assert p.venue is None
    assert p.url is None
    assert p.pdf_url is None
    assert p.doi is None
    assert p.arxiv_id is None
    assert p.citation_source is None
    # Bool/enum defaults
    assert p.is_open_access is None
    assert p.access_status == "unknown"
    # List defaults
    assert p.authors == []
    assert p.sources == []


def test_missing_title_raises():
    """title is required — omitting it must raise ValidationError."""
    with pytest.raises(ValidationError):
        Paper(paper_id="doi:10.1234/test")


def test_invalid_access_status_raises():
    """access_status must be one of the three allowed literals."""
    with pytest.raises(ValidationError):
        Paper(paper_id="doi:10.1234/test", title="T", access_status="maybe")


def test_authors_list_not_shared():
    """Two Paper objects must not share the same default authors list."""
    p1 = Paper(paper_id="id:1", title="A")
    p2 = Paper(paper_id="id:2", title="B")
    p1.authors.append("Alice")
    assert "Alice" not in p2.authors


# --- make_paper_id ---

def test_make_paper_id_doi_priority():
    """DOI takes priority even when other IDs are provided."""
    pid = make_paper_id(doi="10.1234/abc", arxiv_id="2301.00001")
    assert pid == "doi:10.1234/abc"


def test_make_paper_id_doi_lowercase():
    """DOI is lowercased."""
    assert make_paper_id(doi="10.1234/ABC") == "doi:10.1234/abc"


def test_make_paper_id_doi_strips_url_prefix():
    """Common DOI URL prefixes are stripped."""
    assert make_paper_id(doi="https://doi.org/10.1234/X") == "doi:10.1234/x"
    assert make_paper_id(doi="http://doi.org/10.5678/Y") == "doi:10.5678/y"


def test_make_paper_id_arxiv():
    """arXiv ID used when DOI is absent."""
    assert make_paper_id(arxiv_id="2301.00001") == "arxiv:2301.00001"


def test_make_paper_id_s2():
    """Semantic Scholar ID used when DOI and arXiv are absent."""
    assert make_paper_id(semantic_scholar_id="abc123") == "s2:abc123"


def test_make_paper_id_openalex():
    """OpenAlex ID used as last resort."""
    assert make_paper_id(openalex_id="W123") == "openalex:W123"


def test_make_paper_id_none_raises():
    """All None must raise ValueError."""
    with pytest.raises(ValueError):
        make_paper_id()


# --- Round-trip ---

def test_paper_roundtrip():
    """Paper survives model_dump() -> Paper(**data) with identical values."""
    original = Paper(
        paper_id="doi:10.1234/test",
        title="Round Trip",
        abstract="An abstract.",
        year=2024,
        citation_count=42,
        authors=["Alice", "Bob"],
        is_open_access=True,
        access_status="open",
        doi="10.1234/test",
        sources=["semantic_scholar"],
    )
    data = original.model_dump()
    rebuilt = Paper(**data)
    assert rebuilt == original
