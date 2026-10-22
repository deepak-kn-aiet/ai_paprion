"""Unit tests for SQLite search and citation response cache."""

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from data import cache
from data.search import search
from shared.models import Paper

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide a clean temporary SQLite database path for isolated testing."""
    db_file = tmp_path / "test_cache.db"
    yield db_file
    if db_file.exists():
        try:
            db_file.unlink()
        except OSError:
            pass


def _make_paper(**overrides) -> Paper:
    """Helper to construct a Paper object with sensible defaults."""
    base = {
        "paper_id": "doi:10.1038/nature12345",
        "doi": "10.1038/nature12345",
        "title": "Cached Paper Title",
        "abstract": "An abstract for caching test.",
        "year": 2023,
        "venue": "Nature",
        "citation_count": 42,
        "citation_source": "semantic_scholar",
        "authors": ["Alice Scientist"],
        "is_open_access": True,
        "access_status": "open",
        "sources": ["semantic_scholar", "openalex"],
    }
    base.update(overrides)
    return Paper(**base)


# ---------------------------------------------------------------------------
# Query Key Tests
# ---------------------------------------------------------------------------

class TestCacheKeyGeneration:

    def test_deterministic_keys(self):
        """Identical parameters produce identical SHA-256 keys."""
        k1 = cache.make_query_key("deep learning", limit=10, offset=0, providers=["openalex"])
        k2 = cache.make_query_key("deep learning", limit=10, offset=0, providers=["openalex"])
        assert k1 == k2

    def test_query_normalization(self):
        """Casing and excessive whitespace are normalized to the same key."""
        k1 = cache.make_query_key("Deep Learning  ", limit=10)
        k2 = cache.make_query_key("  deep   learning", limit=10)
        assert k1 == k2

    def test_parameter_separation(self):
        """Different limits, offsets, providers, or enrichment produce distinct keys."""
        base_key = cache.make_query_key("query", limit=10, offset=0, providers=["openalex"], enrich_access=False)

        assert base_key != cache.make_query_key("other query", limit=10, offset=0, providers=["openalex"])
        assert base_key != cache.make_query_key("query", limit=20, offset=0, providers=["openalex"])
        assert base_key != cache.make_query_key("query", limit=10, offset=10, providers=["openalex"])
        assert base_key != cache.make_query_key("query", limit=10, offset=0, providers=["semantic_scholar"])
        assert base_key != cache.make_query_key("query", limit=10, offset=0, providers=["openalex"], enrich_access=True)


# ---------------------------------------------------------------------------
# Insertion, Retrieval & Expiry Tests
# ---------------------------------------------------------------------------

class TestCacheStorageAndExpiry:

    def test_cache_miss_on_empty(self, temp_db):
        """Empty cache returns None on lookup."""
        result = cache.get_cached_search("nonexistent_key", db_path=temp_db)
        assert result is None

    def test_insert_and_retrieve_papers(self, temp_db):
        """Papers can be inserted and retrieved with identical fields."""
        paper = _make_paper()
        key = "sample_key_123"

        success = cache.set_cached_search(key, "my query", [paper], db_path=temp_db)
        assert success is True

        cached_papers = cache.get_cached_search(key, db_path=temp_db)
        assert cached_papers is not None
        assert len(cached_papers) == 1
        p = cached_papers[0]
        assert p.paper_id == paper.paper_id
        assert p.title == paper.title
        assert p.doi == paper.doi
        assert p.year == 2023
        assert p.citation_count == 42
        assert p.authors == ["Alice Scientist"]
        assert p.access_status == "open"
        assert p.sources == ["semantic_scholar", "openalex"]

    def test_cache_miss_after_2_day_ttl(self, temp_db):
        """Entries older than 2-day TTL return None (cache miss)."""
        paper = _make_paper()
        key = "expiring_key"

        cache.set_cached_search(key, "query", [paper], db_path=temp_db)

        # Immediate lookup succeeds
        assert cache.get_cached_search(key, db_path=temp_db) is not None

        # Backdate created_at in the database by 3 days (greater than 2-day TTL)
        three_days_ago = time.time() - (3 * 86400)
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE search_cache SET created_at = ? WHERE cache_key = ?", (three_days_ago, key))
        conn.commit()
        conn.close()

        # Lookup should now expire and return None
        expired_result = cache.get_cached_search(key, ttl=cache.DEFAULT_TTL, db_path=temp_db)
        assert expired_result is None

    def test_clear_cache(self, temp_db):
        """clear_cache removes all entries."""
        cache.set_cached_search("k1", "q1", [_make_paper()], db_path=temp_db)
        assert cache.get_cached_search("k1", db_path=temp_db) is not None

        cleared = cache.clear_cache(db_path=temp_db)
        assert cleared is True
        assert cache.get_cached_search("k1", db_path=temp_db) is None


# ---------------------------------------------------------------------------
# Resilience & Error Handling Tests
# ---------------------------------------------------------------------------

class TestCacheResilience:

    def test_corrupted_json_in_cache(self, temp_db):
        """Corrupted JSON payload in database returns None without raising."""
        key = "corrupt_key"
        conn = cache.get_db_connection(temp_db)
        conn.execute(
            "INSERT INTO search_cache (cache_key, query, results_json, created_at) VALUES (?, ?, ?, ?)",
            (key, "query", "INVALID_JSON{{[", time.time()),
        )
        conn.commit()
        conn.close()

        result = cache.get_cached_search(key, db_path=temp_db)
        assert result is None

    def test_database_connection_failure_handled(self):
        """Database connection or operational error returns None/False gracefully."""
        with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("Unable to open database")):
            # Get returns None gracefully
            assert cache.get_cached_search("key") is None

            # Set returns False gracefully
            assert cache.set_cached_search("key", "query", [_make_paper()]) is False


# ---------------------------------------------------------------------------
# Search Pipeline Integration Tests
# ---------------------------------------------------------------------------

class TestSearchPipelineCacheIntegration:

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_second_search_hits_cache_without_calling_providers(self, mock_s2, mock_oa):
        """Subsequent search with identical query hits cache and bypasses external providers."""
        paper = _make_paper(title="Cached Neural Net Paper")
        mock_s2.return_value = [paper]
        mock_oa.return_value = []

        # First call: hits provider and caches
        res1 = search("neural networks", limit=5, use_cache=True)
        assert len(res1) == 1
        assert res1[0].title == "Cached Neural Net Paper"
        assert mock_s2.call_count == 1

        # Reset mock call counter
        mock_s2.reset_mock()
        mock_oa.reset_mock()

        # Second call: hits cache, providers not called
        res2 = search("neural networks", limit=5, use_cache=True)
        assert len(res2) == 1
        assert res2[0].title == "Cached Neural Net Paper"
        assert mock_s2.call_count == 0
        assert mock_oa.call_count == 0

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_provider_failures_are_not_cached_as_empty_results(self, mock_s2, mock_oa):
        """If providers fail, empty results must NOT be saved to cache."""
        mock_s2.side_effect = RuntimeError("S2 down")
        mock_oa.side_effect = RuntimeError("OA down")

        # First call fails: returns []
        res1 = search("failing query", limit=5, use_cache=True)
        assert res1 == []

        # Now suppose providers recover
        mock_s2.side_effect = None
        mock_s2.return_value = [_make_paper(title="Recovered Paper")]
        mock_oa.side_effect = None
        mock_oa.return_value = []

        # Second call should query providers because failure was not cached
        res2 = search("failing query", limit=5, use_cache=True)
        assert len(res2) == 1
        assert res2[0].title == "Recovered Paper"

    @patch("data.search.cache.get_cached_search")
    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_cache_failure_still_allows_provider_search(self, mock_s2, mock_oa, mock_get_cache):
        """If cache raises an unexpected exception, search continues to query providers."""
        mock_get_cache.side_effect = sqlite3.DatabaseError("Disk I/O error")
        mock_s2.return_value = [_make_paper(title="Paper Despite Cache Error")]
        mock_oa.return_value = []

        papers = search("query", limit=5, use_cache=True)
        assert len(papers) == 1
        assert papers[0].title == "Paper Despite Cache Error"

    @patch("data.search.openalex.search")
    @patch("data.search.semantic_scholar.search")
    def test_use_cache_false_bypasses_cache(self, mock_s2, mock_oa):
        """Setting use_cache=False forces fresh provider lookup."""
        mock_s2.return_value = [_make_paper(title="Fresh Paper")]
        mock_oa.return_value = []

        # Populate cache
        search("bypass test", limit=5, use_cache=True)
        assert mock_s2.call_count == 1

        mock_s2.reset_mock()
        mock_s2.return_value = [_make_paper(title="Fresh Paper 2")]

        # Calling with use_cache=False should query provider again
        res = search("bypass test", limit=5, use_cache=False)
        assert mock_s2.call_count == 1
        assert res[0].title == "Fresh Paper 2"
