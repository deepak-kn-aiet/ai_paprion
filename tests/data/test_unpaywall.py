"""Tests for the Unpaywall provider adapter and availability resolver (all mocked)."""

from unittest.mock import MagicMock, patch

import requests

from data.providers import unpaywall
from shared.models import Paper

# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, json_data=None, text=""):
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


def _make_paper(**overrides) -> Paper:
    """Helper to construct a Paper object with sensible defaults."""
    base = {
        "paper_id": "doi:10.1038/nature12373",
        "doi": "10.1038/nature12373",
        "title": "A Great Scientific Paper",
        "access_status": "unknown",
        "is_open_access": None,
        "pdf_url": None,
        "sources": ["semantic_scholar"],
    }
    base.update(overrides)
    return Paper(**base)


# ---------------------------------------------------------------------------
# Open Access & Paywalled Paper Resolution
# ---------------------------------------------------------------------------

class TestAvailabilityResolution:

    def test_open_access_paper_resolution(self):
        """Open-access response sets status to 'open' and updates pdf_url."""
        api_data = {
            "doi": "10.1038/nature12373",
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://www.nature.com/articles/nature12373.pdf",
                "url": "https://doi.org/10.1038/nature12373",
            },
        }
        paper = _make_paper()

        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(200, api_data)

            enriched = unpaywall.enrich_paper(paper)

        assert enriched.is_open_access is True
        assert enriched.access_status == "open"
        assert enriched.pdf_url == "https://www.nature.com/articles/nature12373.pdf"
        assert "unpaywall" in enriched.sources

    def test_paywalled_paper_resolution(self):
        """Paywalled paper response sets status to 'paywalled' and is_open_access=False."""
        api_data = {
            "doi": "10.1000/182",
            "is_oa": False,
            "best_oa_location": None,
        }
        paper = _make_paper(doi="10.1000/182")

        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(200, api_data)

            enriched = unpaywall.enrich_paper(paper)

        assert enriched.is_open_access is False
        assert enriched.access_status == "paywalled"
        assert enriched.pdf_url is None
        assert "unpaywall" in enriched.sources

    def test_fallback_to_oa_locations_list(self):
        """If best_oa_location has no pdf, check oa_locations list."""
        api_data = {
            "doi": "10.1038/test",
            "is_oa": True,
            "best_oa_location": {"url": "https://example.com/landing", "url_for_pdf": None},
            "oa_locations": [
                {"url_for_pdf": "https://repo.example.edu/paper.pdf"},
            ],
        }
        paper = _make_paper(doi="10.1038/test")

        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(200, api_data)

            enriched = unpaywall.enrich_paper(paper)

        assert enriched.pdf_url == "https://repo.example.edu/paper.pdf"

    def test_preserves_existing_pdf_url_if_unpaywall_lacks_one(self):
        """An existing valid pdf_url is never overwritten with None/empty."""
        api_data = {
            "doi": "10.1038/test",
            "is_oa": True,
            "best_oa_location": None,
            "oa_locations": [],
        }
        paper = _make_paper(pdf_url="https://arxiv.org/pdf/1706.03762")

        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(200, api_data)

            enriched = unpaywall.enrich_paper(paper)

        assert enriched.pdf_url == "https://arxiv.org/pdf/1706.03762"
        assert enriched.access_status == "open"


# ---------------------------------------------------------------------------
# Missing Fields & Configuration
# ---------------------------------------------------------------------------

class TestMissingFieldsAndConfig:

    def test_missing_doi_skipped(self):
        """Paper without a DOI is returned unchanged without network calls."""
        paper = _make_paper(doi=None, paper_id="openalex:W123")
        with patch("data.providers.unpaywall.requests.get") as mock_get:
            enriched = unpaywall.enrich_paper(paper)
            assert enriched == paper
            assert not mock_get.called

    def test_missing_unpaywall_email(self):
        """When UNPAYWALL_EMAIL is None, paper is returned unchanged and access_status remains unknown."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = None
            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"
            assert not mock_get.called

    def test_empty_string_unpaywall_email(self):
        """When UNPAYWALL_EMAIL is an empty string, lookup is skipped."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "   "
            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"
            assert not mock_get.called


# ---------------------------------------------------------------------------
# HTTP Statuses & Error Handling
# ---------------------------------------------------------------------------

class TestHttpAndNetworkErrors:

    def test_404_not_found(self):
        """HTTP 404 (DOI not in Unpaywall) returns paper unchanged."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(404, text="DOI not found")

            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"
            assert "unpaywall" not in enriched.sources

    def test_422_unprocessable_doi(self):
        """HTTP 422 (malformed DOI format) returns paper unchanged."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(422, text="Unprocessable DOI")

            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"

    def test_connection_error(self):
        """Network connection exception returns paper unchanged without crashing."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"

    def test_timeout_error(self):
        """Request timeout returns paper unchanged without crashing."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.side_effect = requests.Timeout("Request timed out")

            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"

    def test_invalid_json(self):
        """Invalid JSON response returns paper unchanged."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            resp = _mock_response(200)
            resp.json.side_effect = ValueError("Corrupt JSON")
            mock_get.return_value = resp

            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"


# ---------------------------------------------------------------------------
# Rate Limiting & Retry Backoff
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def test_429_retries_then_succeeds(self):
        """HTTP 429 followed by 200 enriches the paper successfully."""
        paper = _make_paper()
        api_data = {"is_oa": True, "best_oa_location": {"url_for_pdf": "https://oa.pdf"}}

        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.time.sleep"), \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.side_effect = [
                _mock_response(429),
                _mock_response(200, api_data),
            ]

            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "open"
            assert enriched.pdf_url == "https://oa.pdf"

    def test_429_exhausts_retries(self):
        """Exhausting all 429 retries returns paper unchanged."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.time.sleep"), \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(429)

            enriched = unpaywall.enrich_paper(paper)
            assert enriched.access_status == "unknown"
            assert mock_get.call_count == unpaywall._MAX_RETRIES

    def test_429_exponential_backoff_timing(self):
        """Verifies sleep intervals double: 1.0, 2.0, 4.0."""
        paper = _make_paper()
        with patch("data.providers.unpaywall.requests.get") as mock_get, \
             patch("data.providers.unpaywall.time.sleep") as mock_sleep, \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"
            mock_get.return_value = _mock_response(429)

            unpaywall.enrich_paper(paper)
            sleep_delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert sleep_delays == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Batch Enrichment
# ---------------------------------------------------------------------------

class TestBatchEnrichment:

    def test_enrich_papers_batch(self):
        """Enriches list of papers, preserving order and handling mixed DOIs."""
        p1 = _make_paper(paper_id="doi:10.1/a", doi="10.1/a")
        p2 = _make_paper(paper_id="openalex:W2", doi=None)  # No DOI
        p3 = _make_paper(paper_id="doi:10.1/c", doi="10.1/c")

        def fake_get(url, **kwargs):
            if "10.1/a" in url:
                return _mock_response(200, {"is_oa": True, "best_oa_location": {"url_for_pdf": "https://a.pdf"}})
            if "10.1/c" in url:
                return _mock_response(200, {"is_oa": False, "best_oa_location": None})
            return _mock_response(404)

        with patch("data.providers.unpaywall.requests.get", side_effect=fake_get), \
             patch("data.providers.unpaywall.config") as mock_cfg:
            mock_cfg.UNPAYWALL_EMAIL = "test@example.com"

            enriched_list = unpaywall.enrich_papers([p1, p2, p3])

        assert len(enriched_list) == 3
        # p1 open
        assert enriched_list[0].access_status == "open"
        assert enriched_list[0].pdf_url == "https://a.pdf"
        # p2 unchanged
        assert enriched_list[1].access_status == "unknown"
        assert enriched_list[1].pdf_url is None
        # p3 paywalled
        assert enriched_list[2].access_status == "paywalled"

    def test_enrich_empty_list_returns_empty(self):
        """enrich_papers([]) cleanly returns []."""
        assert unpaywall.enrich_papers([]) == []
