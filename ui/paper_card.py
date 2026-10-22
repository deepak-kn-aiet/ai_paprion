"""Backward-compatibility shim for paper card UI components."""

from frontend.ui.paper_card import (
    render_metric_cards,
    render_paper_card,
    render_score_breakdown,
)

__all__ = [
    "render_metric_cards",
    "render_paper_card",
    "render_score_breakdown",
]
