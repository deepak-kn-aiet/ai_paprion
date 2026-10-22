"""Frontend UI package for AI Research Workspace."""

from __future__ import annotations

from .dashboard import (
    EXAMPLE_TOPICS,
    render_initial_dashboard,
)
from .diagnostics import (
    render_planned_queries,
    render_score_analysis,
)
from .footer import render_footer
from .formatters import (
    COMPONENT_KEY_ALIASES,
    SCORE_BREAKDOWN_NOTE,
    build_paper_card,
    build_paper_overview,
    build_paper_verdict,
    build_score_breakdown,
    compute_weighted_score,
    format_authors,
    format_count,
    format_score,
    is_valid_component_score,
    to_progress_value,
)
from .header import (
    render_hero,
    render_report_header,
)
from .hero_shader import render_hero_shader
from .metrics import render_research_metrics
from .paper_card import (
    render_metric_cards,
    render_paper_card,
    render_score_breakdown,
)
from .paper_library import render_paper_library
from .particles import render_particle_background
from .sidebar import (
    SIDEBAR_NAV_ITEMS,
    record_recent_search,
    render_sidebar,
)
from .theme import (
    THEME_CSS_PATH,
    load_theme_css,
    render_theme,
)
from .tubes_cursor import render_tubes_cursor
from .verdict import (
    format_synthesis_citations,
    render_executive_overview,
    render_verdict_section,
)

__all__ = [
    "COMPONENT_KEY_ALIASES",
    "EXAMPLE_TOPICS",
    "SCORE_BREAKDOWN_NOTE",
    "SIDEBAR_NAV_ITEMS",
    "THEME_CSS_PATH",
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
    "load_theme_css",
    "record_recent_search",
    "render_executive_overview",
    "render_footer",
    "render_hero",
    "render_hero_shader",
    "render_initial_dashboard",
    "render_metric_cards",
    "render_paper_card",
    "render_paper_library",
    "render_particle_background",
    "render_planned_queries",
    "render_report_header",
    "render_research_metrics",
    "render_score_analysis",
    "render_score_breakdown",
    "render_sidebar",
    "render_theme",
    "render_tubes_cursor",
    "render_verdict_section",
    "to_progress_value",
]
