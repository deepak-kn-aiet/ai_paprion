"""Central config — reads configuration from environment variables or Streamlit secrets."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env if it exists, no error if missing


def get_config(key: str, default: str | None = None) -> str | None:
    """Retrieve a configuration string from os.environ or Streamlit secrets.

    Priority:
    1. os.environ / local .env
    2. Streamlit Community Cloud secrets (st.secrets)
    3. default fallback value
    """
    val = os.getenv(key)
    if val is not None and val != "":
        return val

    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            secret_val = st.secrets[key]
            if secret_val is not None and str(secret_val) != "":
                return str(secret_val)
    except Exception:  # noqa: BLE001, S110 - graceful fallback when streamlit is not running or secrets.toml is absent
        pass

    return default


def get_bool_config(key: str, default: bool = False) -> bool:
    """Retrieve a boolean configuration flag from os.environ or Streamlit secrets."""
    val = get_config(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")


def get_int_config(key: str, default: int) -> int:
    """Retrieve an integer configuration value from os.environ or Streamlit secrets."""
    val = get_config(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


SEMANTIC_SCHOLAR_API_KEY: str | None = get_config("SEMANTIC_SCHOLAR_API_KEY")
OPENALEX_EMAIL: str | None = get_config("OPENALEX_EMAIL")
OPENALEX_API_KEY: str | None = get_config("OPENALEX_API_KEY")
UNPAYWALL_EMAIL: str | None = get_config("UNPAYWALL_EMAIL")

# LLM / Gemini provider settings (Member B)
GEMINI_API_KEY: str | None = get_config("GEMINI_API_KEY")
LLM_MODEL: str = get_config("LLM_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
LLM_PROVIDER: str = (get_config("LLM_PROVIDER", "gemini") or "gemini").strip().lower()

# Groq provider settings
GROQ_API_KEY: str | None = get_config("GROQ_API_KEY")
GROQ_MODEL: str = get_config("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile"

# Query Planning & API Quota Reduction Settings
PLANNER_MODE: str = (get_config("PLANNER_MODE", "llm") or "llm").strip().lower()
USE_DETERMINISTIC_PLANNER: bool = (
    get_bool_config("USE_DETERMINISTIC_PLANNER", default=False)
    or PLANNER_MODE == "deterministic"
)
SYNTHESIS_MAX_PAPERS: int = get_int_config("SYNTHESIS_MAX_PAPERS", default=10)
SYNTHESIS_MAX_ABSTRACT_CHARS: int = get_int_config("SYNTHESIS_MAX_ABSTRACT_CHARS", default=1000)


