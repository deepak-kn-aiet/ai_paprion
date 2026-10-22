"""Interactive Terminal Verification Tool for Member A (Data & Retrieval Module).

Provides live API diagnostic checks, unified search inspection, deduplication analysis,
cache hit/miss/expiry simulation, and Unpaywall OA resolution testing.
"""

import argparse
import copy
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data import cache
from data.providers import openalex, semantic_scholar, unpaywall
from data.search import deduplicate_papers, search
from shared.models import Paper

# ANSI Color codes for readable terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _fmt(val, default: str = "[Not Available]") -> str:
    """Format an optional value, cleanly indicating missing/empty values."""
    if val is None:
        return f"{DIM}{default}{RESET}"
    if isinstance(val, str) and not val.strip():
        return f"{DIM}{default}{RESET}"
    if isinstance(val, list) and not val:
        return f"{DIM}{default}{RESET}"
    return str(val)


def print_header(title: str) -> None:
    """Print a section header."""
    bar = "=" * 70
    print(f"\n{CYAN}{BOLD}{bar}\n  {title}\n{bar}{RESET}\n")


def print_paper_card(paper: Paper, index: int | None = None) -> None:
    """Render a structured, readable view of a single Paper object."""
    idx_str = f"#{index} " if index is not None else ""
    print(f"{GREEN}{BOLD}--- {idx_str}{paper.title} ---{RESET}")
    print(f"  {BOLD}Canonical ID:{RESET}  {paper.paper_id}")
    print(f"  {BOLD}Year / Venue:{RESET}  {_fmt(paper.year)} | {_fmt(paper.venue)}")
    
    # Authors
    authors_str = ", ".join(paper.authors) if paper.authors else "[No authors listed]"
    print(f"  {BOLD}Authors:{RESET}       {authors_str}")

    # Identifiers
    id_parts = []
    if paper.doi:
        id_parts.append(f"DOI: {paper.doi}")
    if paper.arxiv_id:
        id_parts.append(f"arXiv: {paper.arxiv_id}")
    if paper.semantic_scholar_id:
        id_parts.append(f"S2: {paper.semantic_scholar_id}")
    if paper.openalex_id:
        id_parts.append(f"OpenAlex: {paper.openalex_id}")
    ids_display = " | ".join(id_parts) if id_parts else "[No provider identifiers]"
    print(f"  {BOLD}Identifiers:{RESET}   {ids_display}")

    # URLs
    print(f"  {BOLD}URL:{RESET}           {_fmt(paper.url)}")
    print(f"  {BOLD}PDF URL:{RESET}       {_fmt(paper.pdf_url)}")

    # Citations & Availability
    cit_count = _fmt(paper.citation_count)
    cit_src = f" (via {paper.citation_source})" if paper.citation_source else ""
    influential = f" (influential: {paper.influential_citation_count})" if paper.influential_citation_count is not None else ""
    print(f"  {BOLD}Citations:{RESET}     {cit_count}{cit_src}{influential}")

    status_color = GREEN if paper.access_status == "open" else (RED if paper.access_status == "paywalled" else YELLOW)
    oa_flag = f" (is_oa={paper.is_open_access})" if paper.is_open_access is not None else ""
    print(f"  {BOLD}Access:{RESET}        {status_color}{paper.access_status}{oa_flag}{RESET}")
    print(f"  {BOLD}Sources:{RESET}       {', '.join(paper.sources) if paper.sources else '[None]'}")

    # Abstract snippet
    if paper.abstract:
        snippet = paper.abstract.strip()
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        print(f"  {BOLD}Abstract:{RESET}      {snippet}")
    else:
        print(f"  {BOLD}Abstract:{RESET}      {DIM}[No abstract provided]{RESET}")
    print()


def show_environment_status() -> None:
    """Safely check environment variables without printing secrets."""
    print_header("Environment & Provider Credentials Status")
    
    def check_var(name: str, val: str | None) -> str:
        if val and val.strip():
            # Show presence without leaking the secret
            masked = f"{val[:3]}...{val[-3:]}" if len(val) >= 8 else "***"
            return f"{GREEN}Configured{RESET} ({masked})"
        return f"{YELLOW}Not set{RESET} (will use free/unauthenticated tier)"

    print(f"  Semantic Scholar API Key: {check_var('SEMANTIC_SCHOLAR_API_KEY', config.SEMANTIC_SCHOLAR_API_KEY)}")
    print(f"  OpenAlex Polite Email:    {check_var('OPENALEX_EMAIL', config.OPENALEX_EMAIL)}")
    print(f"  OpenAlex API Key:         {check_var('OPENALEX_API_KEY', config.OPENALEX_API_KEY)}")
    print(f"  Unpaywall Contact Email:  {check_var('UNPAYWALL_EMAIL', config.UNPAYWALL_EMAIL)}")
    print(f"  Active SQLite Cache DB:   {cache.DEFAULT_DB_PATH.resolve()}")
    db_exists = cache.DEFAULT_DB_PATH.exists()
    print(f"  Cache DB File Exists:     {GREEN if db_exists else YELLOW}{db_exists}{RESET}")


def run_single_provider_test(provider: str, query: str = "transformers", limit: int = 5) -> None:
    """Query a single provider, measure response time, and show sample diagnostics."""
    print_header(f"Live Provider Test: {provider.upper()}")
    print(f"Searching for: '{query}' (limit={limit})")
    
    t0 = time.time()
    papers: list[Paper] = []
    error_msg = None

    try:
        if provider == "semantic_scholar":
            papers = semantic_scholar.search(query, limit=limit)
        elif provider == "openalex":
            papers = openalex.search(query, limit=limit)
        else:
            print(f"{RED}Unknown provider: {provider}{RESET}")
            return
    except Exception as exc:
        error_msg = str(exc)

    duration = time.time() - t0

    if error_msg:
        print(f"{RED}Request failed after {duration:.2f}s: {error_msg}{RESET}")
        return

    print(f"{GREEN}Returned {len(papers)} papers in {duration:.2f}s{RESET}\n")
    for i, p in enumerate(papers, start=1):
        print_paper_card(p, index=i)


def run_unpaywall_test(doi: str = "10.1038/nature12373") -> None:
    """Test Unpaywall availability resolution for a specific DOI."""
    print_header(f"Live Unpaywall DOI Test: {doi}")
    if not config.UNPAYWALL_EMAIL:
        print(f"{YELLOW}Warning: UNPAYWALL_EMAIL is not configured. Unpaywall requires an email.{RESET}")

    dummy_paper = Paper(
        paper_id=f"doi:{doi.lower()}",
        doi=doi,
        title="Unpaywall Resolution Probe",
    )
    print(f"Querying Unpaywall API for DOI: {doi}...")
    t0 = time.time()
    enriched = unpaywall.enrich_paper(dummy_paper)
    duration = time.time() - t0

    print(f"Completed in {duration:.2f}s.")
    print(f"  Access Status: {GREEN if enriched.access_status == 'open' else YELLOW}{enriched.access_status}{RESET}")
    print(f"  Is Open Access: {_fmt(enriched.is_open_access)}")
    print(f"  Resolved PDF URL: {_fmt(enriched.pdf_url)}")
    print(f"  Sources: {enriched.sources}")


def run_pipeline_test(
    query: str,
    limit: int = 10,
    offset: int = 0,
    providers: list[str] | None = None,
    enrich_access: bool = False,
    use_cache: bool = True,
    inspect_dedup: bool = True,
) -> None:
    """Run full unified search with diagnostics, breakdown, and deduplication stats."""
    print_header("Unified Retrieval Pipeline Execution")
    print(f"Query:          '{query}'")
    print(f"Limit / Offset: {limit} / {offset}")
    print(f"Providers:      {providers if providers else 'All supported (Semantic Scholar, OpenAlex)'}")
    print(f"Use Cache:      {use_cache}")
    print(f"Enrich Access:  {enrich_access}")

    # Inspect cache directly before executing
    active_providers = providers if providers is not None else ["semantic_scholar", "openalex"]
    clean_q = " ".join(query.strip().lower().split())
    cache_key = cache.make_query_key(clean_q, limit=limit, offset=offset, providers=active_providers, enrich_access=enrich_access)
    is_cached_before = cache.get_cached_search(cache_key) is not None if use_cache else False

    if use_cache:
        print(f"Cache Status:   {GREEN if is_cached_before else YELLOW}{'CACHE HIT EXPECTED' if is_cached_before else 'CACHE MISS EXPECTED'}{RESET}")

    # Measure unified search
    t0 = time.time()
    results = search(
        query=query,
        limit=limit,
        offset=offset,
        providers=providers,
        enrich_access=enrich_access,
        use_cache=use_cache,
    )
    duration = time.time() - t0

    print(f"\n{BOLD}Pipeline completed in {duration:.3f}s. Returned {len(results)} papers.{RESET}")

    # If deduplication inspection requested and cache was missed, show individual counts
    if inspect_dedup and not is_cached_before:
        print(f"\n{CYAN}{BOLD}--- Provider Breakdown & Deduplication Analysis ---{RESET}")
        try:
            raw_s2 = semantic_scholar.search(query, limit=limit, offset=offset) if "semantic_scholar" in active_providers else []
            raw_oa = openalex.search(query, limit=limit, offset=offset) if "openalex" in active_providers else []
            combined_raw = len(raw_s2) + len(raw_oa)
            deduped_total = len(deduplicate_papers(raw_s2 + raw_oa))
            duplicates_merged = combined_raw - deduped_total
            print(f"  Semantic Scholar raw results: {len(raw_s2)}")
            print(f"  OpenAlex raw results:         {len(raw_oa)}")
            print(f"  Combined raw count:           {combined_raw}")
            print(f"  Duplicates merged:            {GREEN}{duplicates_merged}{RESET}")
            print(f"  Unique records before limit:  {deduped_total}")
            print(f"  Returned after limit({limit}):  {len(results)}")
        except Exception as exc:
            print(f"  (Breakdown analysis skipped: {exc})")

    # Display papers
    print(f"\n{CYAN}{BOLD}--- Retrieved Papers ---{RESET}")
    if not results:
        print(f"{YELLOW}No papers found for this query.{RESET}")
    else:
        for idx, paper in enumerate(results, start=1):
            print_paper_card(paper, index=idx)


def run_cache_benchmark_demo(query: str = "reinforcement learning") -> None:
    """Demonstrate cache miss vs cache hit timing in real time."""
    print_header("Cache Miss vs. Cache Hit Demonstration")
    print(f"Test Query: '{query}'")

    # Step 1: Ensure key is cleared
    key = cache.make_query_key(query, limit=5, offset=0, providers=["semantic_scholar", "openalex"], enrich_access=False)
    conn = cache.get_db_connection()
    conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
    conn.commit()
    conn.close()

    print("Step 1: Cache cleared for test key.")
    print("Executing Request #1 (Expect LIVE API fetch & Cache Miss)...")
    t0 = time.time()
    res1 = search(query, limit=5, use_cache=True)
    t1 = time.time() - t0
    print(f"{YELLOW}-> Call #1 took: {t1:.3f}s (Fetched {len(res1)} papers from live APIs){RESET}")

    print("\nExecuting Request #2 with identical parameters (Expect CACHE HIT)...")
    t0 = time.time()
    res2 = search(query, limit=5, use_cache=True)
    t2 = time.time() - t0
    print(f"{GREEN}-> Call #2 took: {t2:.4f}s (Served directly from local SQLite cache){RESET}")

    if t1 > 0:
        speedup = t1 / max(t2, 0.0001)
        print(f"\n{BOLD}Result: Cache hit was {speedup:.1f}x faster than live retrieval.{RESET}")


def run_cache_expiry_simulation(tmp_db_path: Path) -> None:
    """Demonstrate 2-day TTL expiry without deleting production data or waiting 2 days."""
    print_header("Cache 2-Day TTL Expiration Simulation (Isolated Database)")
    print(f"Using isolated temporary test database: {tmp_db_path}")

    paper = Paper(
        paper_id="doi:10.1145/test_expiry",
        doi="10.1145/test_expiry",
        title="Time Capsule Paper on Expiry",
    )
    test_key = "simulation_key_ttl"

    # Step 1: Store entry
    cache.set_cached_search(test_key, "simulation query", [paper], db_path=tmp_db_path)
    print("1. Inserted record into isolated test cache.")

    # Step 2: Immediate check -> HIT
    immediate = cache.get_cached_search(test_key, db_path=tmp_db_path)
    print(f"2. Immediate lookup status: {GREEN}{'CACHE HIT' if immediate else 'MISS'}{RESET}")

    # Step 3: Fast-forward timestamp by 3 days (3 * 86,400s) > 2 days
    three_days_seconds = 3 * 86400
    expired_timestamp = time.time() - three_days_seconds
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("UPDATE search_cache SET created_at = ? WHERE cache_key = ?", (expired_timestamp, test_key))
    conn.commit()
    conn.close()
    print("3. Simulated time advancement: backdated record by 3 days.")

    # Step 4: Lookup -> Expired (MISS)
    expired_check = cache.get_cached_search(test_key, ttl=cache.DEFAULT_TTL, db_path=tmp_db_path)
    if expired_check is None:
        print(f"4. Post-TTL lookup: {GREEN}CACHE MISS (Entry treated as expired after 2 days){RESET}")
    else:
        print(f"4. Post-TTL lookup: {RED}ERROR: Entry did not expire!{RESET}")

    # Clean up temp db
    if tmp_db_path.exists():
        tmp_db_path.unlink()


def run_cache_resilience_simulation(tmp_db_path: Path) -> None:
    """Demonstrate system resilience against a corrupted SQLite cache database."""
    print_header("Cache Error Resilience Simulation (Isolated Corrupted DB)")
    print(f"Writing corrupt payload to isolated database: {tmp_db_path}")

    # Write corrupt non-JSON data into the table
    conn = cache.get_db_connection(tmp_db_path)
    conn.execute(
        "INSERT OR REPLACE INTO search_cache (cache_key, query, results_json, created_at) VALUES (?, ?, ?, ?)",
        ("corrupt_key", "query", "INVALID_MALFORMED_DATA_NOT_JSON", time.time()),
    )
    conn.commit()
    conn.close()

    print("1. Database contains unparseable/corrupted payload.")
    lookup = cache.get_cached_search("corrupt_key", db_path=tmp_db_path)
    print(f"2. Cache retrieval result: {GREEN}{lookup} (Handled gracefully without crashing){RESET}")

    if tmp_db_path.exists():
        tmp_db_path.unlink()


def interactive_menu() -> None:
    """Run interactive terminal menu."""
    while True:
        print_header("Member A: Data & Retrieval Manual Testing Console")
        print("  1. Run unified search across all providers (Interactive)")
        print("  2. Test Semantic Scholar API directly")
        print("  3. Test OpenAlex API directly")
        print("  4. Test Unpaywall DOI availability resolution")
        print("  5. Run Cache Hit / Miss Benchmark demonstration")
        print("  6. Run isolated 7-Day Cache TTL Expiry simulation")
        print("  7. Run isolated Cache Storage Failure / Resilience simulation")
        print("  8. Show current environment credentials status")
        print("  9. Run automated mocked test suite (pytest)")
        print("  0. Exit")
        
        try:
            choice = input(f"\n{BOLD}Select an option [0-9]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "0":
            print("Goodbye.")
            break
        elif choice == "1":
            q = input("Enter search query [default: 'transformer attention']: ").strip() or "transformer attention"
            limit_in = input("Enter result limit [default: 5]: ").strip() or "5"
            limit = int(limit_in) if limit_in.isdigit() else 5
            
            prov_in = input("Select providers: (1) All, (2) Semantic Scholar only, (3) OpenAlex only [default: 1]: ").strip()
            if prov_in == "2":
                provs = ["semantic_scholar"]
            elif prov_in == "3":
                provs = ["openalex"]
            else:
                provs = None

            enrich_in = input("Enable Unpaywall OA enrichment? (y/N): ").strip().lower()
            enrich = enrich_in in ("y", "yes")

            cache_in = input("Enable SQLite cache? (Y/n): ").strip().lower()
            use_cache = cache_in not in ("n", "no")

            run_pipeline_test(
                query=q,
                limit=limit,
                offset=0,
                providers=provs,
                enrich_access=enrich,
                use_cache=use_cache,
                inspect_dedup=True,
            )
        elif choice == "2":
            q = input("Query for Semantic Scholar [default: 'graph neural networks']: ").strip() or "graph neural networks"
            run_single_provider_test("semantic_scholar", query=q, limit=5)
        elif choice == "3":
            q = input("Query for OpenAlex [default: 'quantum computing algorithms']: ").strip() or "quantum computing algorithms"
            run_single_provider_test("openalex", query=q, limit=5)
        elif choice == "4":
            doi = input("Enter DOI to test with Unpaywall [default: 10.1038/nature12373]: ").strip() or "10.1038/nature12373"
            run_unpaywall_test(doi=doi)
        elif choice == "5":
            q = input("Query for cache benchmark [default: 'diffusion models']: ").strip() or "diffusion models"
            run_cache_benchmark_demo(query=q)
        elif choice == "6":
            tmp = PROJECT_ROOT / ".cache" / "simulation_ttl_test.db"
            run_cache_expiry_simulation(tmp)
        elif choice == "7":
            tmp = PROJECT_ROOT / ".cache" / "simulation_corrupt_test.db"
            run_cache_resilience_simulation(tmp)
        elif choice == "8":
            show_environment_status()
        elif choice == "9":
            print_header("Executing Mocked Unit Test Suite")
            import pytest
            pytest.main(["-v", str(PROJECT_ROOT / "tests")])
        else:
            print(f"{RED}Invalid option. Please choose between 0 and 9.{RESET}")

        input(f"\n{DIM}Press Enter to return to main menu...{RESET}")


def main() -> None:
    """CLI entrypoint supporting both interactive mode and command line flags."""
    parser = argparse.ArgumentParser(description="Member A: Data & Retrieval Manual Test Tool")
    parser.add_argument("--query", "-q", type=str, help="Search query to run non-interactively")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Number of papers to fetch")
    parser.add_argument("--offset", type=int, default=0, help="Result offset")
    parser.add_argument("--providers", "-p", nargs="+", choices=["semantic_scholar", "openalex"], help="Providers to query")
    parser.add_argument("--enrich", action="store_true", help="Enable Unpaywall open access enrichment")
    parser.add_argument("--no-cache", action="store_true", help="Bypass SQLite cache")
    parser.add_argument("--benchmark", action="store_true", help="Run cache benchmark test")
    parser.add_argument("--status", action="store_true", help="Show environment and credential status")

    args = parser.parse_args()

    if args.status:
        show_environment_status()
    elif args.benchmark:
        run_cache_benchmark_demo(args.query or "deep reinforcement learning")
    elif args.query:
        run_pipeline_test(
            query=args.query,
            limit=args.limit,
            offset=args.offset,
            providers=args.providers,
            enrich_access=args.enrich,
            use_cache=not args.no_cache,
            inspect_dedup=True,
        )
    else:
        # Default to interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
