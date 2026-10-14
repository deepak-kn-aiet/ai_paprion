"""Tests for unified search aggregator and deduplication service (all mocked)."""

from unittest.mock import patch

import pytest

from data.cache import clear_cache
from data.search import deduplicate_papers, merge_papers, normalize_arxiv_id, normalize_doi, normalize_title, search, unified_search
from shared.models import Paper


@pytest.fixture(autouse=True)
def _reset_search_cache():
    """Ensure search tests run with a clean cache state."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_paper(**overrides) -> Paper:
    """Helper to construct a Paper model for testing."""
    base = {
        "paper_id": "openalex:W100",
        "title": "Default Test Title",
        "abstract": "A default abstract.",
        "year": 2021,
        "venue": "Default Venue",
        "url": "https://example.com/paper",
        "pdf_url": "https://example.com/paper.pdf",
        "citation_count": 10,
        "authors": ["Alice"],
        "is_open_access": True,
        "access_status": "open",
        "sources": ["openalex"],
    }
    base.update(overrides)
    return Paper(**base)


# ---------------------------------------------------------------------------
# Normalization Helper Tests
# ---------------------------------------------------------------------------

class TestNormalizationHelpers:

    def test_normalize_doi(self):
        assert normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
        assert normalize_doi("http://doi.org/10.1234/abc") == "10.1234/abc"
        assert normalize_doi("doi:10.1234/abc") == "10.1234/abc"
        assert normalize_doi("10.1234/abc") == "10.1234/abc"
        assert normalize_doi(None) == ""
        assert normalize_doi("") == ""

    def test_normalize_arxiv_id(self):
        assert normalize_arxiv_id("https://arxiv.org/abs/1706.03762v2") == "1706.03762"
        assert normalize_arxiv_id("http://arxiv.org/abs/1706.03762") == "1706.03762"
        assert normalize_arxiv_id("arxiv:1706.03762v1") == "1706.03762"
        assert normalize_arxiv_id("1706.03762") == "1706.03762"
        assert normalize_arxiv_id(None) == ""

    def test_normalize_title(self):
        assert normalize_title("Attention Is All You Need!") == "attentionisallyouneed"
        assert normalize_title("Attention is all you need.") == "attentionisallyouneed"
        assert normalize_title("<b>Attention</b>: Is All You Need?") == "attentionisallyouneed"
        assert normalize_title("A") == ""  # Too short
        assert normalize_title(None) == ""


# ---------------------------------------------------------------------------
# Deduplication Logic Tests
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_deduplicate_by_doi(self):
        """Papers with identical DOIs (even with prefixes or different casing) are merged."""
        p1 = _make_paper(
            paper_id="doi:10.1234/abc",
            doi="https://doi.org/10.1234/ABC",
            title="Title A",
            citation_count=50,
            sources=["semantic_scholar"],
        )
        p2 = _make_paper(
            paper_id="openalex:W123",
            doi="10.1234/abc",
            title="Title A (slightly diff)",
            citation_count=80,
            sources=["openalex"],
        )
        deduped = deduplicate_papers([p1, p2])
        assert len(deduped) == 1
        assert deduped[0].doi == "https://doi.org/10.1234/ABC"
        assert deduped[0].citation_count == 80
        assert set(deduped[0].sources) == {"semantic_scholar", "openalex"}

    def test_deduplicate_by_arxiv_id(self):
        """When DOI is absent, papers with matching arXiv IDs are merged."""
        p1 = _make_paper(
            paper_id="arxiv:1706.03762",
            doi=None,
            arxiv_id="1706.03762v1",
            title="Attention Is All You Need",
            sources=["semantic_scholar"],
        )
        p2 = _make_paper(
            paper_id="openalex:W999",
            doi=None,
            arxiv_id="https://arxiv.org/abs/1706.03762v2",
            title="Attention Is All You Need",
            sources=["openalex"],
        )
        deduped = deduplicate_papers([p1, p2])
        assert len(deduped) == 1
        assert deduped[0].arxiv_id == "1706.03762v1"
        assert set(deduped[0].sources) == {"semantic_scholar", "openalex"}

    def test_deduplicate_by_title_fallback(self):
        """When both DOI and arXiv are missing, matching normalized titles are merged."""
        p1 = _make_paper(
            paper_id="s2:abc1",
            doi=None,
            arxiv_id=None,
            semantic_scholar_id="abc1",
            title="Deep Residual Learning for Image Recognition",
            sources=["semantic_scholar"],
        )
        p2 = _make_paper(
            paper_id="openalex:W777",
            doi=None,
            arxiv_id=None,
            openalex_id="W777",
            title="Deep Residual Learning for Image Recognition!",
            sources=["openalex"],
        )
        deduped = deduplicate_papers([p1, p2])
        assert len(deduped) == 1
        assert deduped[0].semantic_scholar_id == "abc1"
        assert deduped[0].openalex_id == "W777"
        assert set(deduped[0].sources) == {"semantic_scholar", "openalex"}

    def test_different_dois_not_merged_even_if_title_matches(self):
        """Papers with conflicting DOIs must not be merged even if titles match."""
        p1 = _make_paper(
            paper_id="doi:10.1111/first",
            doi="10.1111/first",
            title="Common Title Across Volumes",
        )
        p2 = _make_paper(
            paper_id="doi:10.2222/second",
            doi="10.2222/second",
            title="Common Title Across Volumes",
        )
        deduped = deduplicate_papers([p1, p2])
        assert len(deduped) == 2

    def test_different_arxiv_ids_not_merged_even_if_title_matches(self):
        """Papers with conflicting arXiv IDs must not be merged even if titles match."""
        p1 = _make_paper(
            paper_id="arxiv:2301.00001",
            doi=None,
            arxiv_id="2301.00001",
            title="Workshop Paper",
        )
        p2 = _make_paper(
            paper_id="arxiv:2301.00002",
            doi=None,
            arxiv_id="2301.00002",
            title="Workshop Paper",
        )
        deduped = deduplicate_papers([p1, p2])
        assert len(deduped) == 2

    def test_order_of_appearance_preserved(self):
        """Distinct papers maintain the order of first appearance."""
        p1 = _make_paper(paper_id="doi:10.1/a", doi="10.1/a", title="Paper 1")
        p2 = _make_paper(paper_id="doi:10.1/b", doi="10.1/b", title="Paper 2")
        p3 = _make_paper(paper_id="doi:10.1/c", doi="10.1/c", title="Paper 3")
        deduped = deduplicate_papers([p1, p2, p3])
        assert [p.paper_id for p in deduped] == ["doi:10.1/a", "doi:10.1/b", "doi:10.1/c"]


# ---------------------------------------------------------------------------
# Metadata Merging Tests
# ---------------------------------------------------------------------------

class TestMetadataMerging:

    def test_merge_preserves_non_empty_values(self):
        """Missing/None values in incoming paper do not overwrite useful base metadata."""
        base = _make_paper(
            paper_id="doi:10.1000/1",
            doi="10.1000/1",
            title="Complete Title",
            abstract="Detailed abstract text.",
            venue="Top Conference",
            year=2022,
            pdf_url="https://example.com/p.pdf",
            citation_count=50,
            influential_citation_count=10,
            authors=["Alice", "Bob"],
            sources=["semantic_scholar"],
        )
        incoming = _make_paper(
            paper_id="openalex:W1",
            openalex_id="W1",
            doi="10.1000/1",
            title="Complete Title",
            abstract=None,
            venue=None,
            year=None,
            pdf_url=None,
            citation_count=None,
            influential_citation_count=None,
            authors=[],
            sources=["openalex"],
        )
        merged = merge_papers(base, incoming)
        assert merged.abstract == "Detailed abstract text."
        assert merged.venue == "Top Conference"
        assert merged.year == 2022
        assert merged.pdf_url == "https://example.com/p.pdf"
        assert merged.citation_count == 50
        assert merged.influential_citation_count == 10
        assert merged.authors == ["Alice", "Bob"]
        assert merged.openalex_id is not None
        assert set(merged.sources) == {"semantic_scholar", "openalex"}

    def test_merge_prefers_longer_abstract(self):
        """When both have abstracts, the more complete/longer one is kept."""
        base = _make_paper(
            paper_id="doi:10.1/test",
            doi="10.1/test",
            abstract="Short summary.",
        )
        incoming = _make_paper(
            paper_id="openalex:W2",
            doi="10.1/test",
            abstract="A much longer and comprehensive abstract explaining the methodology.",
        )
        merged = merge_papers(base, incoming)
        assert merged.abstract == "A much longer and comprehensive abstract explaining the methodology."

    def test_merge_citation_counts_and_source(self):
        """Higher citation count is selected and the citation_source updated."""
        base = _make_paper(
            paper_id="doi:10.1/test",
            doi="10.1/test",
            citation_count=100,
            citation_source="semantic_scholar",
        )
        incoming = _make_paper(
            paper_id="openalex:W3",
            doi="10.1/test",
            citation_count=250,
            citation_source="openalex",
        )
        merged = merge_papers(base, incoming)
        assert merged.citation_count == 250
        assert merged.citation_source == "openalex"

    def test_merge_preserves_all_provider_ids_and_updates_paper_id(self):
        """Provider IDs from both records are preserved and canonical paper_id reflects DOI."""
        base = _make_paper(
            paper_id="openalex:W555",
            doi=None,
            openalex_id="W555",
            title="Same Paper Title",
        )
        incoming = _make_paper(
            paper_id="s2:s2_id_999",
            doi="10.5555/sample",
            arxiv_id="1901.00001",
            semantic_scholar_id="s2_id_999",
            title="Same Paper Title",
        )
        merged = merge_papers(base, incoming)
        assert merged.doi == "10.5555/sample"
        assert merged.arxiv_id == "1901.00001"
        assert merged.semantic_scholar_id == "s2_id_999"
        assert merged.openalex_id == "W555"
        # Canonical paper_id elevates to DOI priority
        assert merged.paper_id == "doi:10.5555/sample"


# ---------------------------------------------------------------------------
# Unified Search Aggregator Tests
# ---------------------------------------------------------------------------

class TestUnifiedSearch:

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_search_calls_both_providers_and_merges(self, mock_s2, mock_oa):
        """Aggregator queries both providers in parallel and deduplicates results."""
        paper_s2 = _make_paper(
            paper_id="doi:10.1234/nlp",
            doi="10.1234/nlp",
            title="Transformer Models",
            citation_count=50,
            sources=["semantic_scholar"],
        )
        paper_oa = _make_paper(
            paper_id="openalex:W888",
            doi="10.1234/nlp",
            title="Transformer Models",
            citation_count=90,
            sources=["openalex"],
        )
        unique_oa = _make_paper(
            paper_id="openalex:W777",
            doi="10.1234/vision",
            title="Vision Models",
            sources=["openalex"],
        )

        mock_s2.return_value = [paper_s2]
        mock_oa.return_value = [paper_oa, unique_oa]

        results = search("transformers", limit=10)

        assert len(results) == 2
        # First paper should be merged
        assert results[0].doi == "10.1234/nlp"
        assert results[0].citation_count == 90
        assert set(results[0].sources) == {"semantic_scholar", "openalex"}
        # Second paper is unique
        assert results[1].doi == "10.1234/vision"

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_one_provider_fails_other_succeeds(self, mock_s2, mock_oa):
        """If Semantic Scholar raises an exception, OpenAlex papers are still returned."""
        mock_s2.side_effect = RuntimeError("S2 service down")
        mock_oa.return_value = [_make_paper(paper_id="openalex:W1", title="Paper from OA")]

        results = search("quantum", limit=5)
        assert len(results) == 1
        assert results[0].title == "Paper from OA"

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_openalex_fails_s2_succeeds(self, mock_s2, mock_oa):
        """If OpenAlex fails, Semantic Scholar papers are still returned."""
        mock_s2.return_value = [_make_paper(paper_id="doi:10.1/s2", doi="10.1/s2", title="Paper from S2")]
        mock_oa.side_effect = Exception("OpenAlex timeout")

        results = search("graphs", limit=5)
        assert len(results) == 1
        assert results[0].title == "Paper from S2"

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_both_providers_fail_returns_empty(self, mock_s2, mock_oa):
        """If all providers fail, returns empty list without crashing."""
        mock_s2.side_effect = Exception("S2 network error")
        mock_oa.side_effect = Exception("OA network error")

        results = search("error query")
        assert results == []

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_empty_results_from_both(self, mock_s2, mock_oa):
        """Both providers returning empty lists yields an empty list."""
        mock_s2.return_value = []
        mock_oa.return_value = []

        results = search("nonexistent topic")
        assert results == []

    def test_empty_query_returns_immediately(self):
        """Empty or whitespace-only query returns [] without making provider calls."""
        with patch("data.search.semantic_scholar.search") as mock_s2, \
             patch("data.search.openalex.search") as mock_oa:
            assert search("") == []
            assert search("   ") == []
            assert not mock_s2.called
            assert not mock_oa.called

    def test_limit_zero_or_negative_returns_empty(self):
        """limit <= 0 returns empty list without calling providers."""
        with patch("data.search.semantic_scholar.search") as mock_s2:
            assert search("test", limit=0) == []
            assert search("test", limit=-5) == []
            assert not mock_s2.called

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_limit_caps_returned_results(self, mock_s2, mock_oa):
        """Aggregator caps total returned deduplicated papers to *limit*."""
        mock_s2.return_value = [
            _make_paper(paper_id=f"doi:10.1/{i}", doi=f"10.1/{i}", title=f"Paper {i}")
            for i in range(5)
        ]
        mock_oa.return_value = [
            _make_paper(paper_id=f"openalex:W{i}", openalex_id=f"W{i}", title=f"OA Paper {i}")
            for i in range(5)
        ]

        results = search("test", limit=3)
        assert len(results) == 3

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_specific_provider_selection(self, mock_s2, mock_oa):
        """Passing providers=['openalex'] only queries OpenAlex."""
        mock_oa.return_value = [_make_paper(paper_id="openalex:W10")]

        results = search("test", providers=["openalex"])
        assert len(results) == 1
        assert not mock_s2.called
        assert mock_oa.called

    def test_unified_search_alias(self):
        """unified_search is an alias for search."""
        assert unified_search is search


# ---------------------------------------------------------------------------
# Retrieval Pipeline Integration Tests (Task 7)
# ---------------------------------------------------------------------------

class TestRetrievalPipelineIntegration:

    @patch("data.search.unpaywall.enrich_papers")
    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_enrichment_disabled_by_default(self, mock_s2, mock_oa, mock_enrich):
        """By default (enrich_access=False), unpaywall.enrich_papers is never called."""
        mock_s2.return_value = [_make_paper(doi="10.1000/1", title="Paper 1")]
        mock_oa.return_value = []

        papers = search("query")
        assert len(papers) == 1
        assert not mock_enrich.called

    @patch("data.search.unpaywall.enrich_papers")
    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_enrichment_called_when_enabled(self, mock_s2, mock_oa, mock_enrich):
        """When enrich_access=True, unpaywall.enrich_papers is called with deduplicated papers."""
        raw_paper = _make_paper(doi="10.1000/1", title="Paper 1")
        enriched_paper = _make_paper(
            doi="10.1000/1",
            title="Paper 1",
            access_status="open",
            pdf_url="https://oa.pdf",
            sources=["semantic_scholar", "unpaywall"],
        )
        mock_s2.return_value = [raw_paper]
        mock_oa.return_value = []
        mock_enrich.return_value = [enriched_paper]

        papers = search("query", enrich_access=True)
        assert len(papers) == 1
        assert mock_enrich.called
        assert papers[0].access_status == "open"
        assert papers[0].pdf_url == "https://oa.pdf"

    @patch("data.search.unpaywall.enrich_papers")
    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_enrichment_happens_after_deduplication(self, mock_s2, mock_oa, mock_enrich):
        """Duplicates from different providers are merged before being sent to Unpaywall."""
        p_s2 = _make_paper(
            paper_id="doi:10.1000/dup",
            doi="10.1000/dup",
            title="Duplicate Paper",
            sources=["semantic_scholar"],
        )
        p_oa = _make_paper(
            paper_id="openalex:W999",
            doi="10.1000/dup",
            title="Duplicate Paper",
            sources=["openalex"],
        )
        mock_s2.return_value = [p_s2]
        mock_oa.return_value = [p_oa]
        mock_enrich.side_effect = lambda papers: papers  # Pass-through

        papers = search("query", enrich_access=True)
        # unpaywall.enrich_papers should receive exactly 1 merged paper, not 2
        assert mock_enrich.called
        passed_papers = mock_enrich.call_args[0][0]
        assert len(passed_papers) == 1
        assert len(papers) == 1
        assert set(papers[0].sources) == {"semantic_scholar", "openalex"}

    @patch("data.search.unpaywall.enrich_papers")
    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_result_order_and_metadata_preserved_during_enrichment(self, mock_s2, mock_oa, mock_enrich):
        """Paper order and metadata remain intact across enrichment."""
        p1 = _make_paper(paper_id="doi:10.1/1", doi="10.1/1", title="First", year=2020)
        p2 = _make_paper(paper_id="doi:10.1/2", doi="10.1/2", title="Second", year=2022)
        mock_s2.return_value = [p1, p2]
        mock_oa.return_value = []

        def fake_enrich(papers):
            return [
                Paper(**{**p.model_dump(), "access_status": "open", "is_open_access": True})
                for p in papers
            ]
        mock_enrich.side_effect = fake_enrich

        papers = search("test", limit=5, enrich_access=True)
        assert len(papers) == 2
        assert [p.title for p in papers] == ["First", "Second"]
        assert papers[0].year == 2020
        assert papers[1].year == 2022
        assert all(p.access_status == "open" for p in papers)

    @patch("data.search.unpaywall.enrich_papers")
    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_unpaywall_failure_does_not_discard_search_results(self, mock_s2, mock_oa, mock_enrich):
        """If Unpaywall raises an unexpected exception, retrieved search results are preserved."""
        p1 = _make_paper(paper_id="doi:10.1/resilient", doi="10.1/resilient", title="Resilient Paper")
        mock_s2.return_value = [p1]
        mock_oa.return_value = []
        mock_enrich.side_effect = RuntimeError("Unpaywall unexpected crash")

        papers = search("query", enrich_access=True)
        assert len(papers) == 1
        assert papers[0].title == "Resilient Paper"
        assert papers[0].doi == "10.1/resilient"

    @patch("data.search.unpaywall.enrich_papers")
    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_enrichment_with_empty_results_and_missing_dois(self, mock_s2, mock_oa, mock_enrich):
        """Empty search results do not invoke enrichment; papers without DOIs pass through safely."""
        # Empty case
        mock_s2.return_value = []
        mock_oa.return_value = []
        assert search("none", enrich_access=True) == []
        assert not mock_enrich.called

        # Missing DOI case
        p_no_doi = _make_paper(paper_id="arxiv:2301.00001", doi=None, arxiv_id="2301.00001")
        mock_s2.return_value = [p_no_doi]
        mock_enrich.side_effect = lambda papers: papers

        papers = search("arxiv", enrich_access=True)
        assert len(papers) == 1
        assert papers[0].arxiv_id == "2301.00001"
