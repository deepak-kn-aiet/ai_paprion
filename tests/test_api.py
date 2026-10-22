"""Tests for the FastAPI application: CORS behavior and the search endpoint.

All provider calls are mocked so no live API requests are made.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from shared.models import Paper

client = TestClient(app)


def _make_paper(**overrides) -> Paper:
    base = {
        "paper_id": "doi:10.1234/test",
        "doi": "10.1234/test",
        "title": "Test Paper",
        "abstract": "An abstract.",
        "year": 2024,
        "citation_count": 10,
        "authors": ["Alice"],
        "sources": ["semantic_scholar"],
    }
    base.update(overrides)
    return Paper(**base)


# ---------------------------------------------------------------------------
# CORS: allowed origins
# ---------------------------------------------------------------------------

class TestCORSAllowedOrigins:

    def test_allowed_origin_receives_cors_headers_on_get(self):
        with patch("main.unified_search", return_value=[_make_paper()]):
            resp = client.get(
                "/api/search?query=transformers",
                headers={"Origin": "http://localhost:5173"},
            )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_allowed_origin_127_receives_cors_headers(self):
        with patch("main.unified_search", return_value=[]):
            resp = client.get(
                "/api/search?query=ai",
                headers={"Origin": "http://127.0.0.1:5173"},
            )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

    def test_disallowed_origin_does_not_receive_cors_headers(self):
        with patch("main.unified_search", return_value=[]):
            resp = client.get(
                "/api/search?query=ai",
                headers={"Origin": "http://evil.example.com"},
            )
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers


# ---------------------------------------------------------------------------
# CORS: preflight OPTIONS
# ---------------------------------------------------------------------------

class TestCORSPreflight:

    def test_preflight_from_allowed_origin_succeeds(self):
        resp = client.options(
            "/api/search",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert "GET" in resp.headers["access-control-allow-methods"].split(", ")
        assert "content-type" in resp.headers["access-control-allow-headers"].lower()

    def test_preflight_from_disallowed_origin_is_rejected(self):
        resp = client.options(
            "/api/search",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert "access-control-allow-origin" not in resp.headers


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

class TestSearchEndpoint:

    def test_search_returns_papers(self):
        paper = _make_paper()
        with patch("main.unified_search", return_value=[paper]) as mock_search:
            resp = client.get("/api/search?query=transformers&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["paper_id"] == "doi:10.1234/test"
        assert data[0]["title"] == "Test Paper"
        mock_search.assert_called_once_with(
            query="transformers",
            limit=5,
            offset=0,
            providers=None,
            enrich_access=False,
            use_cache=True,
        )

    def test_search_passes_all_parameters(self):
        with patch("main.unified_search", return_value=[]) as mock_search:
            client.get(
                "/api/search?query=ai&limit=20&offset=10"
                "&providers=openalex&enrich_access=true&use_cache=false"
            )
        mock_search.assert_called_once_with(
            query="ai",
            limit=20,
            offset=10,
            providers=["openalex"],
            enrich_access=True,
            use_cache=False,
        )

    def test_search_requires_query(self):
        resp = client.get("/api/search")
        assert resp.status_code == 422

    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
