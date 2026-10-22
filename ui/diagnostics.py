"""Backward-compatibility shim for diagnostics and score analysis UI components."""

from frontend.ui.diagnostics import (
    render_planned_queries,
    render_score_analysis,
)

__all__ = [
    "render_planned_queries",
    "render_score_analysis",
]
