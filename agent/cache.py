"""SQLite-backed research synthesis and agent result cache with TTL.

Provides deterministic caching of completed, successful research syntheses
keyed by normalized topic, LLM provider, model, and planner mode.
Never caches failures or empty results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import ResearchAgentResult

logger = logging.getLogger(__name__)

DEFAULT_AGENT_CACHE_DIR = Path(".cache")
DEFAULT_AGENT_DB_PATH = DEFAULT_AGENT_CACHE_DIR / "research_cache.db"
DEFAULT_AGENT_CACHE_TTL = 2 * 86400  # 2 days in seconds


def get_agent_cache_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create or connect to the agent SQLite cache database and initialize schema."""
    path = Path(db_path) if db_path is not None else DEFAULT_AGENT_DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_cache (
            cache_key TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def make_research_cache_key(
    topic: str,
    provider: str = "gemini",
    model: str = "default",
    deterministic: bool = False,
) -> str:
    """Build a deterministic SHA-256 cache key from research configuration."""
    clean_topic = " ".join(topic.strip().lower().split()) if topic else ""
    raw = f"{clean_topic}|{provider.strip().lower()}|{model.strip().lower()}|{deterministic}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_research_result(
    cache_key: str,
    ttl: float = DEFAULT_AGENT_CACHE_TTL,
    db_path: Path | str | None = None,
) -> ResearchAgentResult | None:
    """Retrieve cached research result if present, unexpired, and valid."""
    from agent.agent import ResearchAgentResult

    conn = get_agent_cache_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT result_json, created_at FROM research_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        result_json, created_at = row
        if time.time() - float(created_at) > ttl:
            conn.execute("DELETE FROM research_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
            return None

        data = json.loads(result_json)
        result = ResearchAgentResult(**data)
        result.is_cached = True
        return result
    except Exception as exc:  # noqa: BLE001 - resilience to cache read failure
        logger.warning("Error reading from research cache (key: %s): %s", cache_key, exc)
        return None
    finally:
        conn.close()


def cache_research_result(
    cache_key: str,
    result: ResearchAgentResult,
    db_path: Path | str | None = None,
) -> None:
    """Store a successful, non-empty research result in SQLite cache.

    Never caches failures or results with 0 papers.
    """
    if not result.ranked_papers:
        # Never cache empty or failed results
        return

    conn = get_agent_cache_connection(db_path)
    try:
        # Ensure is_cached is false in serialized storage
        data = result.model_dump(mode="json")
        data["is_cached"] = False
        payload = json.dumps(data)
        conn.execute(
            """
            INSERT OR REPLACE INTO research_cache (cache_key, topic, result_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, result.topic, payload, time.time()),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001 - resilience to cache write failure
        logger.warning("Failed to write to research cache (key: %s): %s", cache_key, exc)
    finally:
        conn.close()


def clear_research_cache(db_path: Path | str | None = None) -> None:
    """Clear all records from research cache."""
    conn = get_agent_cache_connection(db_path)
    try:
        conn.execute("DELETE FROM research_cache")
        conn.commit()
    finally:
        conn.close()
