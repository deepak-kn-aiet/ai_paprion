"""Streamlit application entry point for AI Research Workspace.

Re-exports core application logic and UI components from the frontend package
to maintain full backward compatibility with test runners and external scripts.
"""

from __future__ import annotations

from frontend.app import (
    COMPONENT_KEY_ALIASES,
    EXAMPLE_TOPICS,
    MOBILE_NAV_ITEMS,
    SCORE_BREAKDOWN_NOTE,
    SEARCH_SUGGESTIONS,
    SIDEBAR_NAV_ITEMS,
    THEME_CSS_PATH,
    ResearchAgent,
    ResearchAgentError,
    ResearchAgentResult,
    build_paper_card,
    build_paper_overview,
    build_paper_verdict,
    build_score_breakdown,
    compute_weighted_score,
    format_authors,
    format_count,
    format_score,
    format_synthesis_citations,
    is_valid_component_score,
    is_valid_topic,
    load_theme_css,
    record_recent_search,
    render,
    render_executive_overview,
    render_footer,
    render_hero,
    render_hero_shader,
    render_initial_dashboard,
    render_metric_cards,
    render_mobile_navigation,
    render_overview_page,
    render_overview_visualizations,
    render_paper_card,
    render_paper_library,
    render_particle_background,
    render_planned_queries,
    render_report_header,
    render_research_metrics,
    render_score_analysis,
    render_score_breakdown,
    render_sidebar,
    render_team_banner,
    render_theme,
    render_tubes_cursor,
    render_verdict_section,
    to_progress_value,
)

__all__ = [
    "COMPONENT_KEY_ALIASES",
    "EXAMPLE_TOPICS",
    "MOBILE_NAV_ITEMS",
    "SCORE_BREAKDOWN_NOTE",
    "SEARCH_SUGGESTIONS",
    "SIDEBAR_NAV_ITEMS",
    "THEME_CSS_PATH",
    "ResearchAgent",
    "ResearchAgentError",
    "ResearchAgentResult",
    "build_paper_card",
    "build_paper_overview",
    "build_paper_verdict",
    "build_score_breakdown",
    "compute_weighted_score",
    "format_authors",
    "format_count",
    "format_score",
    "format_synthesis_citations",
    "is_valid_component_score",
    "is_valid_topic",
    "load_theme_css",
    "record_recent_search",
    "render",
    "render_executive_overview",
    "render_footer",
    "render_hero",
    "render_hero_shader",
    "render_initial_dashboard",
    "render_metric_cards",
    "render_mobile_navigation",
    "render_overview_page",
    "render_overview_visualizations",
    "render_paper_card",
    "render_paper_library",
    "render_particle_background",
    "render_planned_queries",
    "render_report_header",
    "render_research_metrics",
    "render_score_analysis",
    "render_score_breakdown",
    "render_sidebar",
    "render_team_banner",
    "render_theme",
    "render_tubes_cursor",
    "render_verdict_section",
    "run_agent_search",
    "to_progress_value",
]

import functools

import agent.agent
from data.search import search


def run_agent_search(topic: str) -> ResearchAgentResult:
    """Execute end-to-end research orchestration using OpenAlex and Gemini."""
    clean_topic = topic.strip()
    if not clean_topic:
        raise ValueError("Topic must be a non-empty string.")
    openalex_search = functools.partial(search, providers=["openalex"])
    agent_cls = globals().get("ResearchAgent") or getattr(agent.agent, "ResearchAgent", ResearchAgent)
    agent_instance = agent_cls(search_fn=openalex_search)
    return agent_instance.run(clean_topic)


if __name__ == "__main__":
    render()
