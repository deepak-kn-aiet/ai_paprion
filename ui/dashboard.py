"""Backward-compatibility shim for dashboard UI component."""

from frontend.ui.dashboard import (
    EXAMPLE_TOPICS,
    PAGE_DESCRIPTIONS,
    PAGE_FEATURES,
    render_empty_results_state,
    render_initial_dashboard,
)

__all__ = [
    "EXAMPLE_TOPICS",
    "PAGE_DESCRIPTIONS",
    "PAGE_FEATURES",
    "render_empty_results_state",
    "render_initial_dashboard",
]
