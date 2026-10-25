"""Data formatting and paper card preparation helpers."""

from __future__ import annotations

from typing import Any

from mock_data import (
    SCORE_COMPONENT_LABELS,
    SCORE_COMPONENT_WEIGHTS,
)
from shared.models import Paper

SCORE_BREAKDOWN_NOTE = (
    "Component scores reflect normalized multi-criteria evaluations across "
    "retrieved literature. When a metric is unavailable, its weight is "
    "redistributed proportionally across remaining components."
)

COMPONENT_KEY_ALIASES: dict[str, list[str]] = {
    "citation_impact": ["citations_per_year", "citation_impact"],
    "influential_citation_ratio": ["influential_citation_ratio"],
    "venue_quality": ["venue_quality"],
    "recency": ["recency"],
    "review_score": ["peer_review", "review_score"],
}


def format_authors(paper: Any) -> str:
    """Format authors list into comma-separated string."""
    if isinstance(paper, Paper):
        authors = paper.authors
    elif isinstance(paper, dict):
        authors = paper.get("authors") or []
    else:
        authors = []
    if not authors:
        return "Unknown authors"
    return ", ".join(authors)


def format_count(value: Any) -> str:
    """Format numeric counts with thousands commas."""
    if isinstance(value, bool) or not isinstance(value, int):
        return "N/A"
    return f"{value:,}"


def format_score(score: Any) -> str:
    """Format 0-1 scores as decimal representation."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "N/A"
    return f"{score:.2f} / 1.0"


def to_progress_value(score: Any) -> float | None:
    """Normalize score to 0.0-1.0 float for progress bars."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return max(0.0, min(1.0, float(score)))


def is_valid_component_score(value: Any) -> bool:
    """Check if value is a valid numeric score in 0.0 to 1.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0


def build_score_breakdown(paper: Any) -> list[dict[str, Any]]:
    """Build standardized score component rows with label, score, weight, contribution."""
    if isinstance(paper, Paper):
        components = paper.score_breakdown or {}
    elif isinstance(paper, dict):
        components = paper.get("score_components") or paper.get("score_breakdown") or {}
    else:
        components = {}

    rows = []
    for name, weight in SCORE_COMPONENT_WEIGHTS.items():
        raw_score = None
        for alias in COMPONENT_KEY_ALIASES.get(name, [name]):
            if alias in components and components[alias] is not None:
                raw_score = components[alias]
                break

        if is_valid_component_score(raw_score) and raw_score is not None:
            score = float(raw_score)
            rows.append(
                {
                    "component": SCORE_COMPONENT_LABELS[name],
                    "score": score,
                    "weight": weight,
                    "contribution": weight * score,
                    "available": True,
                }
            )
        else:
            rows.append(
                {
                    "component": SCORE_COMPONENT_LABELS[name],
                    "score": None,
                    "weight": weight,
                    "contribution": None,
                    "available": False,
                }
            )
    return rows


def compute_weighted_score(breakdown_rows: list[dict[str, Any]]) -> float | None:
    """Compute redistributed weighted score across available components."""
    available = [row for row in breakdown_rows if row["available"]]
    if not available:
        return None
    total_weight = sum(row["weight"] for row in available)
    if total_weight <= 0:
        return None
    return sum(row["contribution"] for row in available) / total_weight


def build_paper_overview(paper: Any) -> dict[str, Any]:
    """Extract paper problem, main contribution, and related topics if available."""
    if isinstance(paper, dict):
        overview = paper.get("overview") or {}
        related_topics = overview.get("related_topics") or []
        return {
            "research_problem": overview.get("research_problem"),
            "main_contribution": overview.get("main_contribution"),
            "related_topics": related_topics,
            "available": bool(
                overview.get("research_problem")
                or overview.get("main_contribution")
                or related_topics
            ),
        }
    return {
        "research_problem": None,
        "main_contribution": None,
        "related_topics": [],
        "available": False,
    }


def build_paper_verdict(paper: Any) -> dict[str, Any]:
    """Build paper-level recommendation status and explanation."""
    if isinstance(paper, Paper):
        if paper.is_recommended is not None:
            rec = "Keep this paper" if paper.is_recommended else "Consider another paper"
            exp = (
                f"Paper quality score: {format_score(paper.score)} "
                f"(recommendation threshold: 0.70)."
            )
            return {"recommendation": rec, "explanation": exp, "available": True}
        if paper.score is not None:
            rec = "Evaluated candidate"
            exp = f"Paper quality score: {format_score(paper.score)}."
            return {"recommendation": rec, "explanation": exp, "available": True}
        return {"recommendation": None, "explanation": None, "available": False}

    if isinstance(paper, dict):
        verdict = paper.get("verdict") or {}
        return {
            "recommendation": verdict.get("recommendation"),
            "explanation": verdict.get("explanation"),
            "available": bool(verdict.get("recommendation")),
        }

    return {"recommendation": None, "explanation": None, "available": False}


def build_paper_card(
    paper: Any, topic: str = "", rank: int | str | None = None
) -> dict[str, Any]:
    """Map Paper model or dictionary to template-ready dictionary for display."""
    score_breakdown = build_score_breakdown(paper)

    if isinstance(paper, Paper):
        paper_rank = rank if rank is not None else "-"
        title = paper.title or "Untitled"
        year = paper.year if paper.year is not None else "N/A"
        venue = paper.venue or "N/A"
        citations = format_count(paper.citation_count)
        influential_citations = format_count(paper.influential_citation_count)
        score_str = format_score(paper.score)
        score_val = to_progress_value(paper.score)
        paper_link = paper.url
        pdf_link = paper.pdf_url
        summary = paper.abstract or ""
        paper_id = paper.paper_id
        is_recommended = paper.is_recommended
        is_open_access = paper.is_open_access
        doi = paper.doi
        arxiv_id = paper.arxiv_id
        access_status = paper.access_status
    elif isinstance(paper, dict):
        paper_rank = rank if rank is not None else paper.get("rank", "-")
        title = paper.get("title") or "Untitled"
        year = paper.get("year", "N/A")
        venue = paper.get("venue") or "N/A"
        citations = format_count(paper.get("citations"))
        influential_citations = format_count(paper.get("influential_citations"))
        score_str = format_score(paper.get("score"))
        score_val = to_progress_value(paper.get("score"))
        paper_link = paper.get("url")
        pdf_link = paper.get("pdf_url")
        summary = paper.get("summary") or ""
        paper_id = paper.get("paper_id", "")
        is_recommended = paper.get("is_recommended")
        is_open_access = paper.get("is_open_access")
        doi = paper.get("doi")
        arxiv_id = paper.get("arxiv_id")
        access_status = paper.get("access_status", "unknown")
    else:
        paper_rank = rank if rank is not None else "-"
        title = "Untitled"
        year = "N/A"
        venue = "N/A"
        citations = "N/A"
        influential_citations = "N/A"
        score_str = "N/A"
        score_val = None
        paper_link = None
        pdf_link = None
        summary = ""
        paper_id = ""
        is_recommended = None
        is_open_access = None
        doi = None
        arxiv_id = None
        access_status = "unknown"

    return {
        "rank": paper_rank,
        "title": title,
        "authors": format_authors(paper),
        "year": year,
        "venue": venue,
        "citations": citations,
        "influential_citations": influential_citations,
        "score": score_str,
        "score_value": score_val,
        "paper_link": paper_link,
        "pdf_link": pdf_link,
        "summary": summary,
        "score_breakdown": score_breakdown,
        "weighted_total": compute_weighted_score(score_breakdown),
        "topic": topic,
        "overview": build_paper_overview(paper),
        "verdict": build_paper_verdict(paper),
        "paper_id": paper_id,
        "is_recommended": is_recommended,
        "is_open_access": is_open_access,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "access_status": access_status,
    }
