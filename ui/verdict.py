"""Backward-compatibility shim for verdict UI component."""

from frontend.ui.verdict import (
    format_synthesis_citations,
    render_executive_overview,
    render_verdict_section,
)

__all__ = [
    "format_synthesis_citations",
    "render_executive_overview",
    "render_verdict_section",
]
