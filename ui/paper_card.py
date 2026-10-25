"""Backward-compatibility shim for paper card UI components."""

from frontend.ui.paper_card import (
    SCORE_BREAKDOWN_NOTE,
    render_metric_cards,
    render_paper_card,
    render_score_breakdown,
)

__all__ = [
    "SCORE_BREAKDOWN_NOTE",
    "render_metric_cards",
    "render_paper_card",
    "render_score_breakdown",
]
