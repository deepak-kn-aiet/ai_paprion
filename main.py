"""FastAPI application exposing the unified research-paper retrieval pipeline.

Member A (Data & Retrieval): wraps the existing `data.search.search()` pipeline
and configures CORS so Member C's frontend can call the API from the browser.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from data.search import search as unified_search
from shared.models import Paper

app = FastAPI(
    title="ai_paprion Retrieval API",
    version="0.1.0",
    description="Unified research-paper search across Semantic Scholar and OpenAlex.",
)

FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/search", response_model=list[Paper], tags=["search"])
def search_papers(
    query: str = Query(..., description="Research query string."),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of papers to return."),
    offset: int = Query(0, ge=0, description="Result offset for pagination."),
    providers: list[str] | None = Query(
        None,
        description="Providers to query. Defaults to all supported providers.",
    ),
    enrich_access: bool = Query(
        False,
        description="Enrich results with Unpaywall open-access availability.",
    ),
    use_cache: bool = Query(True, description="Use the local SQLite query cache."),
) -> list[Paper]:
    papers = unified_search(
        query=query,
        limit=limit,
        offset=offset,
        providers=providers,
        enrich_access=enrich_access,
        use_cache=use_cache,
    )
    return papers
