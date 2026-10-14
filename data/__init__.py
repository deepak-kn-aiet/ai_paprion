"""Data package — search, retrieval, and provider adapters."""

from data.cache import clear_cache, get_cached_search, make_query_key, set_cached_search
from data.providers.unpaywall import enrich_paper, enrich_papers
from data.search import deduplicate_papers, merge_papers, search, unified_search

__all__ = [
    "search",
    "unified_search",
    "deduplicate_papers",
    "merge_papers",
    "enrich_paper",
    "enrich_papers",
    "get_cached_search",
    "set_cached_search",
    "clear_cache",
    "make_query_key",
]
