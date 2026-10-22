"""Backward-compatibility shim for dashboard UI component."""

from frontend.ui.dashboard import (
    EXAMPLE_TOPICS,
    render_initial_dashboard,
)

__all__ = ["EXAMPLE_TOPICS", "render_initial_dashboard"]
