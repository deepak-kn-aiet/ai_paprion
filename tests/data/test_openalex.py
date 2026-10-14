"""Tests for the OpenAlex provider (all mocked, no live API calls or keys required)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from data.providers import openalex


# ---------------------------------------------------------------------------
# Sample OpenAlex API fixtures
# ---------------------------------------------------------------------------

def _make_raw_work(**overrides):
    """Build a realistic OpenAlex API work dictionary with sensible defaults."""
    base = {
        "id": "https://openalex.org/W2741809807",
        "title": "Attention Is All You Need",
        "display_name": "Attention Is All You Need",
        "publication_year": 2017,
        "cited_by_count": 90000,
        "doi": "https://doi.org/10.5555/3295222.3295349",
        "ids": {
            "openalex": "https://openalex.org/W2741809807",
            "doi": "https://doi.org/10.5555/3295222.3295349",
            "arxiv": "https://arxiv.org/abs/1706.03762",
        },
        "abstract_inverted_index": {
            "The": [0],
            "dominant": [1],
            "sequence": [2],
            "transduction": [3],
            "models": [4],
        },
        "primary_location": {
            "source": {
                "display_name": "Advances in Neural Information Processing Systems",
            },
            "pdf_url": "https://arxiv.org/pdf/1706.03762",
            "landing_page_url": "https://papers.nips.cc/paper/7181-attention-is-all-you-need",
        },
        "open_access": {
            "is_oa": True,
            "oa_status": "gold",
            "oa_url": "https://arxiv.org/pdf/1706.03762",
        },
        "authorships": [
            {
                "author_position": "first",
                "author": {
                    "id": "https://openalex.org/A5023880011",
                    "display_name": "Ashish Vaswani",
                },
            },
            {
                "author_position": "middle",
                "author": {
                    "id": "https://openalex.org/A5043210987",
                    "display_name": "Noam Shazeer",
                },
            },
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

    def test_returns_normalized_papers(self):
        """A normal response with works produces normalized Paper objects."""
        raw1 = _make_raw_work()
        raw2 = _make_raw_work(
            id="https://openalex.org/W1982736450",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            display_name="BERT: Pre-training of Deep Bidirectional Transformers",
            publication_year=2019,
            cited_by_count=45000,
            doi="https://doi.org/10.18653/v1/N19-1423",
            ids={
                "openalex": "https://openalex.org/W1982736450",
                "doi": "https://doi.org/10.18653/v1/N19-1423",
                "arxiv": "https://arxiv.org/abs/1810.04805",
            },
            primary_location={
                "source": {"display_name": "NAACL-HLT"},
                "pdf_url": "https://aclanthology.org/N19-1423.pdf",
            },
            open_access={"is_oa": True, "oa_url": "https://aclanthology.org/N19-1423.pdf"},
            authorships=[
                {"author": {"display_name": "Jacob Devlin"}},
                {"author": {"display_name": "Ming-Wei Chang"}},
            ],
        )
        api_json = {"meta": {"count": 2, "page": 1, "per_page": 10}, "results": [raw1, raw2]}

        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("transformers", limit=10)

        assert len(papers) == 2
        p = papers[0]
        assert p.title == "Attention Is All You Need"
        assert p.paper_id == "doi:10.5555/3295222.3295349"
        assert p.doi == "10.5555/3295222.3295349"
        assert p.arxiv_id == "1706.03762"
        assert p.openalex_id == "W2741809807"
        assert p.year == 2017
        assert p.venue == "Advances in Neural Information Processing Systems"
        assert p.citation_count == 90000
        assert p.influential_citation_count is None  # OpenAlex does not have this
        assert p.abstract == "The dominant sequence transduction models"
        assert p.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert p.is_open_access is True
        assert p.pdf_url == "https://arxiv.org/pdf/1706.03762"
        assert p.access_status == "unknown"
        assert p.score is None
        assert p.citation_source == "openalex"
        assert "openalex" in p.sources

    def test_reconstruct_abstract_inverted_index(self):
        """Abstract inverted index with non-sequential positions is reconstructed in order."""
        raw = _make_raw_work(
            abstract_inverted_index={
                "world!": [2],
                "Hello": [0],
                "beautiful": [1],
            }
        )
        api_json = {"meta": {"count": 1}, "results": [raw]}

        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("test")

        assert len(papers) == 1
        assert papers[0].abstract == "Hello beautiful world!"

    def test_direct_abstract_string_used_if_present(self):
        """If abstract is already provided as plain text, it is used."""
        raw = _make_raw_work(
            abstract="Precomputed abstract string.",
            abstract_inverted_index=None,
        )
        api_json = {"results": [raw]}

        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("test")

        assert papers[0].abstract == "Precomputed abstract string."

    def test_host_venue_fallback(self):
        """If primary_location has no source name, host_venue is used."""
        raw = _make_raw_work(
            primary_location=None,
            host_venue={"display_name": "Journal of Machine Learning Research"},
        )
        api_json = {"results": [raw]}

        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("test")

        assert papers[0].venue == "Journal of Machine Learning Research"

    def test_open_access_url_used_when_no_primary_pdf(self):
        """If primary_location has no pdf_url, open_access.oa_url is used."""
        raw = _make_raw_work(
            primary_location={"source": {"display_name": "Nature"}, "pdf_url": None},
            open_access={"is_oa": True, "oa_url": "https://nature.com/articles/test.pdf"},
        )
        api_json = {"results": [raw]}

        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("test")

        assert papers[0].pdf_url == "https://nature.com/articles/test.pdf"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class TestPagination:

    def test_offset_converts_to_1_indexed_page(self):
        """offset=0 -> page=1; offset=20 with limit=10 -> page=3."""
        api_json = {"results": []}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)

            openalex.search("query", limit=10, offset=0)
            _, kwargs1 = mock_get.call_args
            assert kwargs1["params"]["page"] == 1
            assert kwargs1["params"]["per_page"] == 10

            openalex.search("query", limit=10, offset=20)
            _, kwargs2 = mock_get.call_args
            assert kwargs2["params"]["page"] == 3
            assert kwargs2["params"]["per_page"] == 10

    def test_explicit_page_parameter(self):
        """Passing explicit page takes precedence."""
        api_json = {"results": []}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            openalex.search("query", limit=25, page=4)
            _, kwargs = mock_get.call_args
            assert kwargs["params"]["page"] == 4
            assert kwargs["params"]["per_page"] == 25

    def test_limit_capped_at_200(self):
        """OpenAlex limits per_page to 200."""
        api_json = {"results": []}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            openalex.search("query", limit=500)
            _, kwargs = mock_get.call_args
            assert kwargs["params"]["per_page"] == 200


# ---------------------------------------------------------------------------
# Credentials and headers
# ---------------------------------------------------------------------------

class TestConfigurationAndHeaders:

    def test_polite_pool_email_configured(self):
        """When OPENALEX_EMAIL is set, User-Agent contains polite pool email and mailto param sent."""
        api_json = {"results": []}
        with patch("data.providers.openalex.requests.get") as mock_get, \
             patch("data.providers.openalex.config") as mock_cfg:
            mock_cfg.OPENALEX_EMAIL = "researcher@example.com"
            mock_cfg.OPENALEX_API_KEY = None
            mock_get.return_value = _mock_response(200, api_json)

            openalex.search("query")
            _, kwargs = mock_get.call_args
            assert "mailto:researcher@example.com" in kwargs["headers"]["User-Agent"]
            assert kwargs["params"]["mailto"] == "researcher@example.com"

    def test_api_key_configured(self):
        """When OPENALEX_API_KEY is set, api-key header and query param are sent."""
        api_json = {"results": []}
        with patch("data.providers.openalex.requests.get") as mock_get, \
             patch("data.providers.openalex.config") as mock_cfg:
            mock_cfg.OPENALEX_EMAIL = None
            mock_cfg.OPENALEX_API_KEY = "test-openalex-key"
            mock_get.return_value = _mock_response(200, api_json)

            openalex.search("query")
            _, kwargs = mock_get.call_args
            assert kwargs["headers"]["api-key"] == "test-openalex-key"
            assert kwargs["params"]["api_key"] == "test-openalex-key"

    def test_no_credentials_clean_defaults(self):
        """Without email or key, default User-Agent is sent with no auth params."""
        api_json = {"results": []}
        with patch("data.providers.openalex.requests.get") as mock_get, \
             patch("data.providers.openalex.config") as mock_cfg:
            mock_cfg.OPENALEX_EMAIL = None
            mock_cfg.OPENALEX_API_KEY = None
            mock_get.return_value = _mock_response(200, api_json)

            openalex.search("query")
            _, kwargs = mock_get.call_args
            assert kwargs["headers"]["User-Agent"] == "ai_paprion"
            assert "api-key" not in kwargs["headers"]
            assert "mailto" not in kwargs["params"]
            assert "api_key" not in kwargs["params"]


# ---------------------------------------------------------------------------
# Identifier mapping & priority
# ---------------------------------------------------------------------------

class TestIdentifierMapping:

    def test_doi_priority_over_arxiv_and_openalex(self):
        """Canonical paper_id uses DOI when DOI is present."""
        raw = _make_raw_work()
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert papers[0].paper_id.startswith("doi:")

    def test_arxiv_id_used_when_no_doi(self):
        """Canonical paper_id uses arXiv ID when DOI is missing."""
        raw = _make_raw_work(
            doi=None,
            ids={"openalex": "https://openalex.org/W123", "arxiv": "https://arxiv.org/abs/2301.00001"},
        )
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert papers[0].paper_id == "arxiv:2301.00001"
            assert papers[0].arxiv_id == "2301.00001"

    def test_openalex_id_used_when_no_doi_or_arxiv(self):
        """Canonical paper_id uses OpenAlex ID when DOI and arXiv are missing."""
        raw = _make_raw_work(
            id="https://openalex.org/W987654",
            doi=None,
            ids={"openalex": "https://openalex.org/W987654"},
        )
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert papers[0].paper_id == "openalex:W987654"
            assert papers[0].openalex_id == "W987654"

    def test_missing_all_identifiers_skipped(self):
        """Work with no DOI, arXiv, or OpenAlex ID is safely skipped."""
        raw = _make_raw_work(id=None, doi=None, ids={})
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert papers == []


# ---------------------------------------------------------------------------
# Missing & malformed fields
# ---------------------------------------------------------------------------

class TestMalformedAndMissingFields:

    def test_missing_title_falls_back_to_display_name(self):
        """If title is None, display_name is used."""
        raw = _make_raw_work(title=None, display_name="Fallback Display Name")
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert len(papers) == 1
            assert papers[0].title == "Fallback Display Name"

    def test_missing_both_title_and_display_name_skipped(self):
        """If both title and display_name are missing or whitespace, skip work."""
        raw = _make_raw_work(title=None, display_name="   ")
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert papers == []

    def test_missing_optional_fields_are_none(self):
        """Missing optional fields cleanly default to None."""
        raw = _make_raw_work(
            publication_year=None,
            cited_by_count=None,
            abstract_inverted_index=None,
            primary_location=None,
            open_access=None,
            authorships=None,
        )
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert len(papers) == 1
            p = papers[0]
            assert p.year is None
            assert p.citation_count is None
            assert p.abstract is None
            assert p.venue is None
            assert p.pdf_url is None
            assert p.is_open_access is None
            assert p.authors == []

    def test_authors_filtering_invalid_records(self):
        """Malformed author records in authorships are safely ignored."""
        raw = _make_raw_work(
            authorships=[
                {"author": {"display_name": "Valid Author"}},
                {"author": None},
                {"author": {"display_name": ""}},
                "invalid_item",
            ]
        )
        api_json = {"results": [raw]}
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, api_json)
            papers = openalex.search("query")
            assert papers[0].authors == ["Valid Author"]

    def test_invalid_json_returns_empty_list(self):
        """Non-JSON response returns empty list without raising."""
        with patch("data.providers.openalex.requests.get") as mock_get:
            resp = _mock_response(200)
            resp.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = resp
            assert openalex.search("query") == []


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

class TestEmptyResults:

    def test_no_results_key(self):
        """Response without 'results' key returns empty list."""
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"meta": {"count": 0}})
            assert openalex.search("query") == []

    def test_empty_results_list(self):
        """Response with empty results list returns empty list."""
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"results": []})
            assert openalex.search("query") == []


# ---------------------------------------------------------------------------
# HTTP Errors & Network failures
# ---------------------------------------------------------------------------

class TestHttpErrors:

    def test_500_server_error(self):
        """HTTP 500 returns empty list without crashing."""
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(500, text="Internal Server Error")
            assert openalex.search("query") == []

    def test_400_bad_request(self):
        """HTTP 400 returns empty list."""
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.return_value = _mock_response(400, text="Bad Request")
            assert openalex.search("query") == []

    def test_network_connection_error(self):
        """Network exception returns empty list."""
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("DNS failure")
            assert openalex.search("query") == []

    def test_request_timeout(self):
        """Timeout exception returns empty list."""
        with patch("data.providers.openalex.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("Timed out")
            assert openalex.search("query") == []


# ---------------------------------------------------------------------------
# Rate limiting (429) & Exponential Backoff
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def test_429_retries_then_succeeds(self):
        """On 429 followed by 200, returns successful paper."""
        api_json = {"results": [_make_raw_work()]}
        with patch("data.providers.openalex.requests.get") as mock_get, \
             patch("data.providers.openalex.time.sleep"):
            mock_get.side_effect = [
                _mock_response(429),
                _mock_response(200, api_json),
            ]
            papers = openalex.search("query")
            assert len(papers) == 1

    def test_429_exhausts_retries(self):
        """When 429 persists across all retries, returns empty list."""
        with patch("data.providers.openalex.requests.get") as mock_get, \
             patch("data.providers.openalex.time.sleep"):
            mock_get.return_value = _mock_response(429)
            papers = openalex.search("query")
            assert papers == []
            assert mock_get.call_count == openalex._MAX_RETRIES

    def test_429_exponential_backoff_delays(self):
        """Verifies sleep calls follow 1.0s, 2.0s, 4.0s backoff."""
        with patch("data.providers.openalex.requests.get") as mock_get, \
             patch("data.providers.openalex.time.sleep") as mock_sleep:
            mock_get.return_value = _mock_response(429)
            openalex.search("query")
            sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
            assert sleep_values == [1.0, 2.0, 4.0]
