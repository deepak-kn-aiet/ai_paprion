"""Tests for the Semantic Scholar provider (all mocked, no real API calls)."""

from unittest.mock import MagicMock, patch

import requests

from data.providers import semantic_scholar as s2

# ---------------------------------------------------------------------------
# Sample API response fixtures
# ---------------------------------------------------------------------------

def _make_raw_paper(**overrides):
    """Build a realistic S2 API paper dict with sensible defaults."""
    base = {
        "paperId": "abc123def456",
        "title": "Attention Is All You Need",
        "abstract": "We propose a new architecture...",
        "year": 2017,
        "venue": "NeurIPS",
        "url": "https://www.semanticscholar.org/paper/abc123def456",
        "citationCount": 90000,
        "influentialCitationCount": 5000,
        "isOpenAccess": True,
        "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762"},
        "externalIds": {
            "DOI": "10.5555/3295222.3295349",
            "ArXiv": "1706.03762",
        },
        "authors": [
            {"authorId": "1", "name": "Ashish Vaswani"},
            {"authorId": "2", "name": "Noam Shazeer"},
        ],
    }
    base.update(overrides)
    return base


def _mock_response(status_code=200, json_data=None, text=""):
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


# ---------------------------------------------------------------------------
# Successful search
# ---------------------------------------------------------------------------

class TestSuccessfulSearch:

    def test_returns_papers(self):
        """A normal response with two papers produces two Paper objects."""
        raw1 = _make_raw_paper()
        raw2 = _make_raw_paper(
            paperId="xyz789",
            title="BERT",
            externalIds={"DOI": "10.18653/v1/N19-1423"},
            openAccessPdf=None,
            isOpenAccess=False,
        )
        api_json = {"total": 2, "offset": 0, "data": [raw1, raw2]}

        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = s2.search("transformers", limit=10)

        assert len(papers) == 2
        p = papers[0]
        assert p.title == "Attention Is All You Need"
        assert p.paper_id == "doi:10.5555/3295222.3295349"
        assert p.doi == "10.5555/3295222.3295349"
        assert p.arxiv_id == "1706.03762"
        assert p.semantic_scholar_id == "abc123def456"
        assert p.year == 2017
        assert p.venue == "NeurIPS"
        assert p.citation_count == 90000
        assert p.influential_citation_count == 5000
        assert p.is_open_access is True
        assert p.pdf_url == "https://arxiv.org/pdf/1706.03762"
        assert p.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert p.citation_source == "semantic_scholar"
        assert "semantic_scholar" in p.sources
        # access_status left as "unknown" (availability check fills it later)
        assert p.access_status == "unknown"
        # score is never set by this adapter
        assert p.score is None

    def test_offset_and_limit_passed(self):
        """offset and limit are forwarded to the API."""
        api_json = {"total": 0, "offset": 5, "data": []}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            s2.search("test", limit=3, offset=5)
            _, kwargs = mock_get.call_args
            assert kwargs["params"]["offset"] == 5
            assert kwargs["params"]["limit"] == 3

    def test_limit_capped_at_100(self):
        """The API max is 100; we cap to that."""
        api_json = {"total": 0, "offset": 0, "data": []}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            s2.search("test", limit=500)
            _, kwargs = mock_get.call_args
            assert kwargs["params"]["limit"] == 100

    def test_api_key_header_set(self):
        """When an API key is configured, x-api-key header is sent."""
        api_json = {"total": 0, "offset": 0, "data": []}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get, \
             patch("data.providers.semantic_scholar.config") as mock_cfg:
            mock_cfg.SEMANTIC_SCHOLAR_API_KEY = "test-key-123"
            mock_get.return_value = _mock_response(200, api_json)
            s2.search("test")
            _, kwargs = mock_get.call_args
            assert kwargs["headers"]["x-api-key"] == "test-key-123"

    def test_no_api_key_no_header(self):
        """When API key is None, no x-api-key header is sent."""
        api_json = {"total": 0, "offset": 0, "data": []}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get, \
             patch("data.providers.semantic_scholar.config") as mock_cfg:
            mock_cfg.SEMANTIC_SCHOLAR_API_KEY = None
            mock_get.return_value = _mock_response(200, api_json)
            s2.search("test")
            _, kwargs = mock_get.call_args
            assert "x-api-key" not in kwargs["headers"]


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

class TestEmptyResults:

    def test_no_data_key(self):
        """Response with no 'data' key returns empty list."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"total": 0})
            assert s2.search("nothing") == []

    def test_empty_data_list(self):
        """Response with empty data list returns empty list."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"total": 0, "data": []})
            assert s2.search("nothing") == []


# ---------------------------------------------------------------------------
# Malformed / incomplete records
# ---------------------------------------------------------------------------

class TestMalformedRecords:

    def test_missing_title_skipped(self):
        """A paper without a title is dropped."""
        raw = _make_raw_paper(title=None)
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            assert s2.search("test") == []

    def test_empty_title_skipped(self):
        """A paper with an empty string title is dropped."""
        raw = _make_raw_paper(title="")
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            assert s2.search("test") == []

    def test_missing_all_identifiers_skipped(self):
        """A paper with no DOI, arXiv ID, or S2 ID is dropped."""
        raw = _make_raw_paper(paperId=None, externalIds={})
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            assert s2.search("test") == []

    def test_missing_optional_fields_preserved_as_none(self):
        """A paper missing optional fields still normalizes; fields are None."""
        raw = _make_raw_paper(
            abstract=None,
            year=None,
            venue=None,
            citationCount=None,
            influentialCitationCount=None,
            openAccessPdf=None,
            isOpenAccess=None,
            authors=None,
        )
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = s2.search("test")
            assert len(papers) == 1
            p = papers[0]
            assert p.abstract is None
            assert p.year is None
            assert p.venue is None
            assert p.citation_count is None
            assert p.influential_citation_count is None
            assert p.pdf_url is None
            assert p.is_open_access is None
            assert p.authors == []

    def test_empty_venue_becomes_none(self):
        """S2 returns empty string for missing venue; we normalize to None."""
        raw = _make_raw_paper(venue="")
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = s2.search("test")
            assert papers[0].venue is None

    def test_paper_with_only_s2_id(self):
        """Paper with only an S2 ID (no DOI, no arXiv) still normalizes."""
        raw = _make_raw_paper(externalIds={})
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = s2.search("test")
            assert len(papers) == 1
            assert papers[0].paper_id == "s2:abc123def456"

    def test_doi_priority_over_arxiv_and_s2(self):
        """paper_id uses DOI when all three identifiers are present."""
        raw = _make_raw_paper()  # has DOI, ArXiv, and S2 ID
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = s2.search("test")
            assert papers[0].paper_id.startswith("doi:")

    def test_authors_with_missing_names_filtered(self):
        """Authors missing the name field are dropped from the list."""
        raw = _make_raw_paper(authors=[
            {"authorId": "1", "name": "Alice"},
            {"authorId": "2", "name": None},
            {"authorId": "3"},
        ])
        api_json = {"total": 1, "data": [raw]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = s2.search("test")
            assert papers[0].authors == ["Alice"]

    def test_invalid_json_returns_empty(self):
        """If the response body is not valid JSON, return empty list."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            resp = _mock_response(200)
            resp.json.side_effect = ValueError("bad json")
            mock_get.return_value = resp
            assert s2.search("test") == []


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------

class TestHttpErrors:

    def test_500_returns_empty(self):
        """Server error returns empty list, no crash."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(500, text="Internal Server Error")
            assert s2.search("test") == []

    def test_400_returns_empty(self):
        """Bad request returns empty list."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value = _mock_response(400, text="Bad query")
            assert s2.search("test") == []

    def test_network_exception_returns_empty(self):
        """A network-level exception returns empty list."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("DNS failure")
            assert s2.search("test") == []

    def test_timeout_returns_empty(self):
        """A timeout returns empty list."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("timed out")
            assert s2.search("test") == []


# ---------------------------------------------------------------------------
# Rate limiting (429)
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def test_429_retries_then_succeeds(self):
        """On 429 followed by 200, returns the successful result."""
        api_json = {"total": 1, "data": [_make_raw_paper()]}
        with patch("data.providers.semantic_scholar.requests.get") as mock_get, \
             patch("data.providers.semantic_scholar.time.sleep"):
            mock_get.side_effect = [
                _mock_response(429),
                _mock_response(200, api_json),
            ]
            papers = s2.search("test")
            assert len(papers) == 1

    def test_429_exhausts_retries(self):
        """If all retries are 429, returns empty list."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get, \
             patch("data.providers.semantic_scholar.time.sleep"):
            mock_get.return_value = _mock_response(429)
            assert s2.search("test") == []
            # Should have been called _MAX_RETRIES times
            assert mock_get.call_count == s2._MAX_RETRIES

    def test_429_uses_exponential_backoff(self):
        """Backoff doubles on each 429 retry."""
        with patch("data.providers.semantic_scholar.requests.get") as mock_get, \
             patch("data.providers.semantic_scholar.time.sleep") as mock_sleep:
            mock_get.return_value = _mock_response(429)
            s2.search("test")
            # Verify sleep calls: 1.0, 2.0, 4.0 (for 3 retries)
            sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
            assert sleep_values == [1.0, 2.0, 4.0]
