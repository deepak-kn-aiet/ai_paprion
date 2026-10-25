"""Backward-compatibility shim for diagnostics and score analysis UI components."""

from frontend.ui.diagnostics import (
    COMPONENT_DESCRIPTIONS,
    RECOMMENDATION_THRESHOLD,
    SCORE_COMPONENT_LABELS,
    SCORE_COMPONENT_WEIGHTS,
    render_planned_queries,
    render_score_analysis,
)

__all__ = [
    "COMPONENT_DESCRIPTIONS",
    "RECOMMENDATION_THRESHOLD",
    "SCORE_COMPONENT_LABELS",
    "SCORE_COMPONENT_WEIGHTS",
    "render_planned_queries",
    "render_score_analysis",
]
