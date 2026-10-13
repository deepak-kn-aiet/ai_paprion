"""Central config — the only file that reads environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if it exists, no error if missing

SEMANTIC_SCHOLAR_API_KEY: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
OPENALEX_EMAIL: str | None = os.getenv("OPENALEX_EMAIL")
UNPAYWALL_EMAIL: str | None = os.getenv("UNPAYWALL_EMAIL")
