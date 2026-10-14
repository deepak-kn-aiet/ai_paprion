# ai_paprion — AI Research Paper Retrieval Agent

ai_paprion is a research-paper retrieval system that aggregates results from
multiple academic providers, normalizes them into a shared data model,
deduplicates across providers, enriches open-access availability, and serves
the unified results through a FastAPI API.

This repository currently contains the **Member A (Data & Retrieval)** module:
provider adapters, paper normalization, deduplication, open-access enrichment,
SQLite query caching, and the FastAPI application. Member B (scoring/ranking)
and Member C (frontend UI) work is not included in this branch.

## 1. Project Overview

ai_paprion is intended to become an end-to-end AI research agent that helps
users discover, rank, and explore academic literature. The current scope
implements the **retrieval backbone**:

- Querying **Semantic Scholar** and **OpenAlex** in parallel.
- Normalizing provider-specific responses into a shared **`Paper`** model.
- Deduplicating across providers using DOI, arXiv ID, and normalized title.
- Optionally enriching each paper with **Unpaywall** open-access data.
- Caching query results in a local **SQLite** database with a two-day TTL.
- Exposing the pipeline through a **FastAPI** REST API with Swagger docs and
  CORS configured for the frontend development server.

**Implemented (Member A):** provider adapters, normalization, deduplication,
metadata merging, canonical paper IDs, Unpaywall enrichment, SQLite cache,
FastAPI endpoints, CORS, manual test script, and the automated test suite.

**Planned / not yet in this branch:** paper scoring and ranking (Member B),
frontend UI and agent orchestration (Member C).

## 2. Features

### Provider adapters
- **Semantic Scholar** (`data/providers/semantic_scholar.py`) — uses the Graph
  API `/paper/search` endpoint. Sends `x-api-key` when
  `SEMANTIC_SCHOLAR_API_KEY` is set. Requests title, abstract, year, venue,
  authors, citation counts, external IDs, open-access PDF, and URL.
- **OpenAlex** (`data/providers/openalex.py`) — uses the Works API `/works`
  endpoint. Sends `mailto` (for the polite pool) and `api_key` when configured.
  Reconstructs abstracts from the inverted index and resolves authors, venue,
  DOI, arXiv ID, and OpenAlex ID.

### Unified retrieval and normalization
- `data/search.search()` runs selected providers in parallel via
  `ThreadPoolExecutor`, normalizes each provider's raw response into a `Paper`,
  then combines the lists.

### Shared `Paper` model
- `shared/models.Paper` is a Pydantic v2 `BaseModel` shared by all members.
  Required fields: `paper_id`, `title`. Optional fields include `abstract`,
  `venue`, `url`, `pdf_url`, `year`, `citation_count`,
  `influential_citation_count`, `review_score`, `score` (Member B), `authors`,
  `is_open_access`, `access_status`, `doi`, `arxiv_id`,
  `semantic_scholar_id`, `openalex_id`, `citation_source`, and `sources`.
- `make_paper_id()` derives a canonical `paper_id` with priority
  **DOI > arXiv ID > Semantic Scholar ID > OpenAlex ID**.

### Deduplication
`deduplicate_papers()` matches records in this priority order:
1. **Normalized DOI** (lowercased, URL-prefix stripped).
2. **Normalized arXiv ID** (URL-prefix stripped, version suffix removed) —
   skipped if DOIs explicitly conflict.
3. **Normalized title** (HTML/punctuation stripped, lowercased, ≥3 chars) —
   skipped if DOIs or arXiv IDs explicitly conflict.

Duplicate records are merged via `merge_papers()`, which preserves the richer
metadata (longer abstract, higher citation count, more authors, any
open-access flag) and re-derives the canonical `paper_id`. First-seen order is
preserved.

### Metadata merging and canonical IDs
`merge_papers()` combines provider identifiers, keeps the higher citation count
(crediting the reporting provider via `citation_source`), preferring the longer
abstract, and appending new sources without duplicates.

### Open-access enrichment (optional)
`data/providers/unpaywall.py` resolves `is_open_access`, `access_status`
(`"open"` / `"paywalled"` / `"unknown"`), and a direct `pdf_url` by DOI.
Requires `UNPAYWALL_EMAIL`. Papers without a DOI are skipped. Runs in parallel
for batches via `ThreadPoolExecutor`.

### SQLite query cache
- `data/cache.py` stores serialized `Paper` lists in
  `.cache/query_cache.db`.
- Cache key is a SHA-256 hash of the normalized query, limit, offset, sorted
  provider list, and `enrich_access` flag.
- **TTL: 2 days (172,800 seconds)** — enforced by `DEFAULT_TTL = 2 * 86400`.
- Results are cached only when at least one provider succeeded.
- All cache operations are wrapped in try/except and never raise.

### FastAPI endpoints and CORS
- `main.py` defines `app = FastAPI(...)` with:
  - `GET /health` — returns `{"status": "ok"}`.
  - `GET /api/search` — unified search, returns `list[Paper]`.
- CORS (`CORSMiddleware`) allows `http://localhost:5173` and
  `http://127.0.0.1:5173`, methods `GET`, `POST`, `OPTIONS`, header
  `Content-Type`, with `allow_credentials=False`.

### Error handling, retries, and pagination
- Every provider adapter never raises: network and HTTP errors are logged and
  return `[]`.
- Retry on HTTP 429 (rate limit) with up to 3 attempts and exponential
  backoff (1s, 2s, 4s) for Semantic Scholar, OpenAlex, and Unpaywall.
- 10-second request timeout per call.
- Provider failure isolation: if one provider fails, the other's results are
  still returned and deduplicated.
- Semantic Scholar supports `offset`/`limit` (max 100 per page). OpenAlex
  supports `per_page` (max 200) and `page`; `offset` is converted to a page
  number automatically.

## 3. Architecture

### Project structure

```
ai_paprion/
├── config.py                      # Environment variable loader (only file that reads .env)
├── main.py                        # FastAPI application: /health, /api/search, CORS
├── requirements.txt
├── .env.example                   # Template for provider credentials
├── .gitignore
│
├── shared/
│   └── models.py                  # Paper (Pydantic v2) + make_paper_id()
│
├── data/
│   ├── __init__.py                # Public API re-exports
│   ├── search.py                  # search() pipeline, deduplicate_papers(), merge_papers()
│   ├── cache.py                   # SQLite cache (2-day TTL)
│   └── providers/
│       ├── __init__.py
│       ├── semantic_scholar.py    # Semantic Scholar Graph API adapter
│       ├── openalex.py            # OpenAlex Works API adapter
│       └── unpaywall.py           # Unpaywall OA enrichment by DOI
│
├── scripts/
│   └── manual_retrieval_test.py   # Interactive CLI + CLI flags for manual testing
│
└── tests/
    ├── __init__.py
    ├── test_smoke.py              # Config import smoke test
    ├── test_api.py                # FastAPI CORS and search endpoint tests
    └── data/
        ├── test_models.py
        ├── test_cache.py
        ├── test_search.py
        ├── test_semantic_scholar.py
        ├── test_openalex.py
        └── test_unpaywall.py
```

### Retrieval flow

```mermaid
flowchart LR
    Client[Client / Browser] -->|HTTP request| API[FastAPI app]
    API -->|GET /api/search| Search[data.search.search]
    Search -->|cache check| Cache[(SQLite cache<br/>2-day TTL)]
    Cache -->|hit| Search
    Search -->|miss, parallel| S2[Semantic Scholar adapter]
    Search -->|miss, parallel| OA[OpenAlex adapter]
    S2 --> Paper[Paper records]
    OA --> Paper
    Paper --> Dedup[deduplicate_papers]
    Dedup --> Merge[merge_papers]
    Merge -->|if enrich_access| Unpaywall[Unpaywall enrichment]
    Unpaywall --> Final[Final Paper list]
    Merge -->|if not enrich| Final
    Final -->|save on provider success| Cache
    Final --> API
    API -->|JSON list[Paper]| Client
```

Cache lookup happens **before** provider calls. Enrichment happens **after**
deduplication and limiting. Cache save happens **after** enrichment, only if at
least one provider succeeded.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Runtime (uses `list[str] \| None` and `X \| None` syntax) |
| **FastAPI 0.143** | REST API framework, automatic Swagger/OpenAPI docs |
| **Uvicorn 0.54** | ASGI server for running the FastAPI app |
| **Pydantic 2.14** | Data validation and serialization for the `Paper` model |
| **Requests 2.34** | HTTP client for provider API calls |
| **python-dotenv 1.2** | Loads environment variables from `.env` |
| **SQLite** (stdlib `sqlite3`) | Local query cache storage |
| **pytest 9.1** | Automated test framework |
| **httpx 0.28** | HTTP client used by FastAPI's `TestClient` in tests |

## 5. Prerequisites

- **Windows PowerShell** (5.1 or later).
- **Python 3.11 or newer** — the code uses PEP 604 union syntax
  (`str | None`, `list[str] | None`).
- **Git**.
- Internet access for live provider tests (Semantic Scholar, OpenAlex,
  Unpaywall).
- Optional credentials (see [Environment variables](#7-environment-variables)):
  - `SEMANTIC_SCHOLAR_API_KEY` — raises the rate limit (optional; the free tier works without it).
  - `OPENALEX_EMAIL` — puts you in OpenAlex's polite pool (optional).
  - `OPENALEX_API_KEY` — raises the OpenAlex rate limit (optional).
  - `UNPAYWALL_EMAIL` — **required** only when `enrich_access=true`.

## 6. Installation and Environment Setup

```powershell
git clone https://github.com/deepak-kn-aiet/ai_paprion.git
cd ai_paprion

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Activating the virtual environment makes `.venv`'s Python interpreter and all
installed packages available in the current PowerShell session. Deactivate with
`deactivate` when done.

If PowerShell blocks `Activate.ps1` with a `PSSecurityException`, run the
following **session-only** workaround (it does not change the machine-wide
policy):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 7. Environment Variables

Copy the example file to `.env`:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` with your values. `.env` is listed in `.gitignore` and must
remain untracked — never commit it.

| Variable | Required? | Description |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Semantic Scholar API key for a higher rate limit. |
| `OPENALEX_EMAIL` | Optional | Email for OpenAlex's polite pool. |
| `OPENALEX_API_KEY` | Optional | OpenAlex API key for a higher rate limit. |
| `UNPAYWALL_EMAIL` | Optional (required for enrichment) | Contact email for the Unpaywall API. |

- **Semantic Scholar** and **OpenAlex** work without any credentials on their
  free tiers.
- **Unpaywall** requires an email only when calling `enrich_access=true`.
  Without it, enrichment is skipped and existing metadata is preserved.

## 8. Running the Application

Start the backend from the project root:

```powershell
python -m uvicorn main:app --reload
```

- Run this from the `ai_paprion/` directory so that imports like
  `from data.search import search` resolve correctly.
- `main:app` refers to the `app` object (`FastAPI` instance) in `main.py`.
- The `--reload` flag restarts the server on file changes (use only in
  development).

Once running:

| Endpoint | URL |
|---|---|
| Health check | `http://127.0.0.1:8000/health` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Search API | `http://127.0.0.1:8000/api/search` |

### `GET /api/search`

Query parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | *(required)* | Research query string. |
| `limit` | int | `10` | Max papers to return (1–100). |
| `offset` | int | `0` | Result offset for pagination. |
| `providers` | list[string] | all | One or both of `semantic_scholar`, `openalex`. |
| `enrich_access` | bool | `false` | Enrich with Unpaywall OA data. |
| `use_cache` | bool | `true` | Use the local SQLite cache. |

Example request (URL-encoded):

```
http://127.0.0.1:8000/api/search?query=AI%20for%20disease%20prediction&limit=5
```

Response: a JSON array of `Paper` objects (see [API examples](#10-api-examples)).

## 9. Manual Testing Instructions

1. **Activate the virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Start the API** and leave that terminal running:
   ```powershell
   python -m uvicorn main:app --reload
   ```

3. **Test `/health`** in a browser: open `http://127.0.0.1:8000/health`.
   You should see `{"status":"ok"}`.

4. **Test a real search via Swagger:** open `http://127.0.0.1:8000/docs`,
   expand `GET /api/search`, click **Try it out**, enter a `query`, and click
   **Execute**.

5. **PowerShell `Invoke-RestMethod` example:**
   ```powershell
   $params = @{ query = "AI for disease prediction"; limit = 5 }
   $papers = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/search" -Body $params
   $papers | Select-Object title, year, citation_count, doi
   ```

### Manual retrieval script

The interactive CLI at `scripts/manual_retrieval_test.py` provides live
diagnostics:

```powershell
python scripts/manual_retrieval_test.py
```

Menu options:

| Option | Description |
|---|---|
| 1 | Run unified search across all providers (interactive query, limit, providers, enrichment, cache) |
| 2 | Test Semantic Scholar API directly |
| 3 | Test OpenAlex API directly |
| 4 | Test Unpaywall DOI availability resolution |
| 5 | Run Cache Hit / Miss benchmark demonstration |
| 6 | Run isolated 2-day Cache TTL expiry simulation |
| 7 | Run isolated cache storage failure / resilience simulation |
| 8 | Show environment credentials status (masked) |
| 9 | Run the automated mocked test suite (pytest) |
| 0 | Exit |

CLI flags for non-interactive use:

```powershell
python scripts/manual_retrieval_test.py --query "transformer attention" --limit 5 --enrich
python scripts/manual_retrieval_test.py --status
python scripts/manual_retrieval_test.py --benchmark
```

### Automated tests

Run the full suite (all tests are mocked — no live provider calls):

```powershell
python -m pytest -v
```

Coverage: model normalization, deduplication, merging, cache (set/get/clear/TTL),
provider normalization, Unpaywall enrichment, FastAPI CORS behavior, and the
search endpoint. **133 tests pass** as of this commit.

### Provider, cache, and enrichment testing

- **Provider-specific:** use menu options 2 and 3 (live) or the provider test
  files under `tests/data/` (mocked).
- **Cache hit/miss:** menu option 5 demonstrates that the second identical
  query is served from SQLite and is significantly faster.
- **Cache expiration:** menu option 6 backdates a cache entry by 3 days in an
  isolated database and confirms it is treated as expired (2-day TTL).
- **Cache resilience:** menu option 7 writes corrupted JSON into an isolated
  database and confirms the cache lookup returns `None` without crashing.
- **Access enrichment:** run `enrich_access=true` on the API or use menu option
  4 to resolve a single DOI's OA status via Unpaywall.

Live provider results depend on network availability and provider rate limits.
All provider calls include 429 retries with exponential backoff.

## 10. API Examples

### Request

```
GET /api/search?query=AI%20for%20disease%20prediction&limit=2
```

### Response structure

A JSON array of `Paper` objects:

```json
[
  {
    "paper_id": "doi:10.1038/example",
    "title": "Example Paper Title",
    "abstract": "The abstract text...",
    "venue": "Nature",
    "url": "https://doi.org/10.1038/example",
    "pdf_url": "https://example.com/paper.pdf",
    "year": 2024,
    "citation_count": 150,
    "influential_citation_count": 12,
    "review_score": null,
    "score": null,
    "authors": ["Alice Smith", "Bob Jones"],
    "is_open_access": true,
    "access_status": "open",
    "doi": "10.1038/example",
    "arxiv_id": null,
    "semantic_scholar_id": "abc123",
    "openalex_id": "W12345",
    "citation_source": "semantic_scholar",
    "sources": ["semantic_scholar", "openalex", "unpaywall"]
  }
]
```

### Important fields

| Field | Description |
|---|---|
| `paper_id` | Canonical ID (DOI > arXiv > S2 > OpenAlex). |
| `title` | Paper title. |
| `abstract` | Abstract text (reconstructed from OpenAlex inverted index when needed). |
| `year` | Publication year. |
| `authors` | List of author display names. |
| `doi`, `arxiv_id`, `semantic_scholar_id`, `openalex_id` | Provider identifiers. |
| `url`, `pdf_url` | Landing page and direct PDF URLs. |
| `citation_count`, `influential_citation_count` | Citation metadata. |
| `citation_source` | Provider that reported the citation count. |
| `is_open_access`, `access_status` | Open-access availability (from provider data or Unpaywall). |
| `sources` | List of providers that returned this paper. |
| `score` | Ranking score — `null` until Member B implements scoring. |
