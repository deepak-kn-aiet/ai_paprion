"""Central config — the only file that reads environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env if it exists, no error if missing

SEMANTIC_SCHOLAR_API_KEY: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
OPENALEX_EMAIL: str | None = os.getenv("OPENALEX_EMAIL")
OPENALEX_API_KEY: str | None = os.getenv("OPENALEX_API_KEY")
UNPAYWALL_EMAIL: str | None = os.getenv("UNPAYWALL_EMAIL")

# LLM / Gemini provider settings (Member B)
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# Groq provider settings
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Query Planning & API Quota Reduction Settings
PLANNER_MODE: str = os.getenv("PLANNER_MODE", "llm").strip().lower()
USE_DETERMINISTIC_PLANNER: bool = (
    os.getenv("USE_DETERMINISTIC_PLANNER", "false").strip().lower() in ("true", "1", "yes")
    or PLANNER_MODE == "deterministic"
)
SYNTHESIS_MAX_PAPERS: int = int(os.getenv("SYNTHESIS_MAX_PAPERS", "10"))
SYNTHESIS_MAX_ABSTRACT_CHARS: int = int(os.getenv("SYNTHESIS_MAX_ABSTRACT_CHARS", "1000"))


