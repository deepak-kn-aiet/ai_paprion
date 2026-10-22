"""SQLite-based search and citation response cache with 2-day TTL."""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

from shared.models import Paper

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache")
DEFAULT_DB_PATH = DEFAULT_CACHE_DIR / "query_cache.db"
DEFAULT_TTL = 2 * 86400  # 2 days in seconds (172,800s)


def get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create or connect to SQLite database and initialize schema if needed."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_cache (
            cache_key TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def make_query_key(
    query: str,
    limit: int = 10,
    offset: int = 0,
    providers: list[str] | None = None,
    enrich_access: bool = False,
) -> str:
    """Build a deterministic SHA-256 cache key from search parameters."""
    clean_q = " ".join(query.strip().lower().split()) if query else ""
    sorted_prov = sorted(providers) if providers is not None else ["default"]
    raw = f"{clean_q}|{limit}|{offset}|{','.join(sorted_prov)}|{enrich_access}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_search(
    cache_key: str,
    ttl: float = DEFAULT_TTL,
    db_path: Path | str | None = None,
) -> list[Paper] | None:
    """Retrieve cached search results if present and within TTL.

    Returns list of Paper objects on hit, or None on miss, expiry, or error.
    Never raises exceptions.
    """
    try:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT results_json, created_at FROM search_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            results_json, created_at = row
            # 7-day TTL check
            if time.time() - created_at > ttl:
                logger.debug("Cache entry for key %s expired", cache_key)
                return None

            raw_list = json.loads(results_json)
            if not isinstance(raw_list, list):
                return None

            return [Paper(**item) for item in raw_list]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Cache lookup failed gracefully: %s", exc)
        return None


def set_cached_search(
    cache_key: str,
    query: str,
    papers: list[Paper],
    db_path: Path | str | None = None,
) -> bool:
    """Save normalized search results to cache.

    Returns True on success, False on error. Never raises exceptions.
    """
    try:
        serialized = json.dumps([p.model_dump() for p in papers])
        now = time.time()
        conn = get_db_connection(db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO search_cache (cache_key, query, results_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, query, serialized, now),
            )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Cache write failed gracefully: %s", exc)
        return False


def clear_cache(db_path: Path | str | None = None) -> bool:
    """Clear all records from the search cache.

    Returns True on success, False on error. Never raises exceptions.
    """
    try:
        conn = get_db_connection(db_path)
        try:
            conn.execute("DELETE FROM search_cache")
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Cache clear failed: %s", exc)
        return False
