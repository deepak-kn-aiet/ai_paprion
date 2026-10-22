# ai_paprion — AI Research Agent

**ai_paprion** is an academic research platform and intelligent literature exploration agent. It automates literature discovery, multi-criteria paper quality scoring, and evidence-grounded research synthesis, providing both an interactive Streamlit graphical dashboard and a FastAPI backend service.

---

## 1. Project Overview & Architecture

The project integrates three major modules across team roles:
1. **Literature Retrieval & Deduplication (Member A)**: Unified parallel searching across **OpenAlex** and **Semantic Scholar**, canonical identifier resolution (`make_paper_id`), multi-source deduplication, optional Unpaywall open-access resolution, SQLite response caching (2-day TTL), and a FastAPI REST API (`main.py`).
2. **AI Reasoning & Research Agent Orchestration (Member B)**: LLM query planner (`QueryPlanner`), transparent multi-criteria paper scoring (`score_papers`), grounded research overview and verdict synthesis (`OverviewEngine`), and end-to-end research orchestration (`ResearchAgent`).
3. **Interactive UI & Test Suite (Member C)**: Streamlit web dashboard (`app.py`) featuring modern SaaS lavender styling (`ui/styles.css`), paper cards, expandable metric breakdowns, recent searches, and a 37-test suite.

> **UI Status Notice**
> The Streamlit interface (`app.py`) currently runs in **demonstration mode** using sample records from `mock_data.py` with clearly labeled demonstration disclaimers. End-to-end integration between the Streamlit UI and the live `ResearchAgent` orchestration pipeline is the final milestone.

```
ai_paprion/
├── app.py                         # Streamlit interactive UI entry point
├── main.py                        # FastAPI REST API entry point
├── mock_data.py                   # Demonstration dataset & scoring weights for UI
├── config.py                      # Central environment configuration loader (.env)
├── requirements.txt               # Unified project dependencies
├── .env.example                   # Template for API keys and configuration
├── .gitignore                     # Git exclusion rules
│
├── ui/
│   └── styles.css                 # SaaS workspace lavender theme for Streamlit
│
├── agent/                         # AI Reasoning & Orchestration (Member B)
│   ├── __init__.py                # Package exports
│   ├── planner.py                 # LLM multi-query planner (QueryPlanner, GeminiClient)
│   ├── scoring.py                 # Multi-criteria scoring & ranking (score_papers)
│   ├── overview.py                # Thematic clusters, verdict & calibrated confidence
│   └── agent.py                   # ResearchAgent orchestrator (ResearchAgentResult)
│
├── data/                          # Data & Retrieval Layer (Member A)
│   ├── __init__.py                # Public retrieval exports
│   ├── search.py                  # search(), deduplicate_papers(), merge_papers()
│   ├── cache.py                   # SQLite cache with 2-day TTL (.cache/query_cache.db)
│   └── providers/
│       ├── openalex.py            # OpenAlex Works API adapter (rate limits, backoff)
│       ├── semantic_scholar.py    # Semantic Scholar Graph API adapter
│       └── unpaywall.py           # Unpaywall open-access enrichment
│
├── shared/
│   ├── __init__.py
│   └── models.py                  # Shared Paper schema (Pydantic v2) + make_paper_id()
│
└── tests/                         # Comprehensive automated test suite (287 tests)
    ├── __init__.py
    ├── test_app.py                # Streamlit UI & mock data tests (37 tests)
    ├── test_agent.py              # ResearchAgent orchestration tests
    ├── test_planner.py            # QueryPlanner tests
    ├── test_scoring.py            # Multi-criteria paper scoring tests
    ├── test_overview.py           # OverviewEngine & verdict synthesis tests
    ├── test_api.py                # FastAPI REST endpoint tests
    ├── test_smoke.py              # Configuration import smoke test
    └── data/                      # Retrieval & provider test suite
        ├── test_openalex.py       # OpenAlex mocked provider tests
        ├── test_semantic_scholar.py# Semantic Scholar mocked provider tests
        ├── test_unpaywall.py      # Unpaywall mocked provider tests
        ├── test_search.py         # Unified search & deduplication tests
        ├── test_cache.py          # SQLite query cache tests
        └── test_models.py         # Paper model & make_paper_id tests
```

---

## 2. Key Features

### A. Streamlit Interactive Dashboard (`app.py`)
- **Modern SaaS Aesthetics**: Custom lavender and clean white UI styles with responsive containers and tag rows.
- **Paper Cards**: Visual displays showing publication rank, year, venue, citation metrics, author list, and links to papers or PDFs.
- **Score Breakdown Expanders**: Transparent display of weighted component contributions (citation impact, venue quality, recency, review score, influential citations) with proportional weight redistribution when metrics are missing.
- **Thematic Overviews & Verdicts**: Expandable containers displaying synthesized research problems, contributions, and consensus recommendations.
- **Recent Searches & Progress**: Session-based query history and real-time execution progress indicators.

### B. AI Research Agent (`agent/`)
- **Query Planner (`agent/planner.py`)**: Uses Google Gemini (or mock mode) to analyze a broad research question and generate 3–5 targeted academic queries spanning foundational surveys, technical methodologies, empirical benchmarks, and real-world applications.
- **Multi-Criteria Scoring (`agent/scoring.py`)**: Ranks candidate papers using configurable weighted components: citation impact, influential citation ratio, venue tier quality, recency decay, and peer review score.
- **Overview & Verdict Engine (`agent/overview.py`)**: Synthesizes top-ranked candidate evidence into thematic clusters, key empirical findings, unresolved gaps, calibrated confidence levels, and an executive verdict statement with 100% citation referential integrity.
- **Research Agent Orchestrator (`agent/agent.py`)**: Connects planning, multi-query retrieval, cross-query deduplication, ranking, and synthesis into `ResearchAgent.run()`.

### C. Unified Literature Retrieval (`data/`)
- **Dual Academic Providers**: Executes concurrent searches across **OpenAlex** (`data/providers/openalex.py`) and **Semantic Scholar** (`data/providers/semantic_scholar.py`).
- **Resilience & Backoff**: Handles HTTP 429 rate limits with exponential backoff (1s, 2s, 4s retries), 10s request timeouts, and provider failure isolation.
- **Deduplication & Merging**: Cross-matches records by normalized DOI, arXiv ID, and fuzzy title, preserving the most complete metadata and recomputing canonical IDs via `make_paper_id()`.
- **Open-Access Enrichment**: Resolves OA PDF links via Unpaywall.
- **Local SQLite Cache**: Stores serialized search results for 2 days in `.cache/query_cache.db`.

### D. FastAPI REST Server (`main.py`)
- `GET /health`: Health check endpoint returning `{"status": "ok"}`.
- `GET /api/search`: Unified academic search endpoint with query, limit, offset, provider selection, and open-access enrichment flags. Includes CORS middleware for frontend integration.

---

## 3. Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.11+ | Uses modern typing (`list[str] \| None`, PEP 604) |
| **Frontend UI** | Streamlit (>=1.30.0) | Interactive dashboard interface |
| **Backend REST API** | FastAPI (0.143.1) & Uvicorn (0.54.0) | High-performance async ASGI API service |
| **Data Validation** | Pydantic v2 (2.14.1) | Strongly typed schemas (`Paper`, `ResearchOverview`, `QueryPlan`) |
| **LLM Provider** | Google GenAI SDK (1.60.0) | Gemini models (`gemini-2.5-flash`) for reasoning |
| **HTTP Client** | Requests (2.34.2) & HTTPX (0.28.1) | Provider API requests and FastAPI testing |
| **Environment** | python-dotenv (1.2.4) | Secure `.env` loading |
| **Caching** | SQLite3 (Python standard library) | Local query cache |
| **Testing** | Pytest (9.1.1) | Automated test suite (287 tests) |

---

## 4. Installation & Setup

### Prerequisites
- Python 3.11 or newer.
- Git.
- Internet access for live academic searches and Gemini API queries.

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/deepak-kn-aiet/ai_paprion.git
cd ai_paprion

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Environment Variables & Credentials

Copy the example environment template to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your API keys (the `.env` file is excluded from Git):

```ini
# Academic Retrieval Providers
SEMANTIC_SCHOLAR_API_KEY=   # Optional: increases Semantic Scholar rate limit
OPENALEX_EMAIL=             # Optional: polite pool identification for OpenAlex
OPENALEX_API_KEY=           # Optional: authenticated OpenAlex API key
UNPAYWALL_EMAIL=            # Optional: required only for open-access PDF enrichment

# AI Reasoning & LLM Configuration (Member B)
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.5-flash
LLM_PROVIDER=gemini         # Set to "mock" for offline deterministic testing
```

---

## 6. Running the Applications

### A. Run the Streamlit Dashboard
Launch the interactive web UI:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### B. Run the FastAPI REST Server
Start the backend API server:
```bash
uvicorn main:app --reload --port 8000
```
- API Base: `http://localhost:8000`
- Interactive Swagger UI Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Unified Search: `http://localhost:8000/api/search?query=retrieval+augmented+generation&limit=5`

---

## 7. Running the Automated Test Suite

The repository includes a comprehensive 287-test suite covering the UI, scoring, planning, overview engine, agent orchestration, and provider retrieval:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run UI tests only
pytest tests/test_app.py -v

# Run AI Agent orchestration and logic tests only
pytest tests/test_agent.py tests/test_planner.py tests/test_scoring.py tests/test_overview.py -v

# Run retrieval provider tests only
pytest tests/data/ -v
```

All unit tests run completely offline using mocked network fixtures and do not require live API keys or external network connections.

---

## 8. Team Roles & Contribution Matrix

- **Member A (Data & Retrieval)**: Built the multi-provider retrieval pipeline (`data/providers/`), deduplication, identifier canonicalization, Unpaywall OA resolver, SQLite cache, and FastAPI API (`main.py`).
- **Member B (AI Reasoning & Orchestration)**: Implemented the Gemini query planner (`agent/planner.py`), paper scoring algorithm (`agent/scoring.py`), overview and verdict synthesis engine (`agent/overview.py`), and end-to-end orchestration (`agent/agent.py`).
- **Member C (Frontend UI & Testing)**: Designed and implemented the Streamlit workspace (`app.py`), custom theme (`ui/styles.css`), paper cards, demonstration dataset (`mock_data.py`), and automated UI tests (`tests/test_app.py`).
