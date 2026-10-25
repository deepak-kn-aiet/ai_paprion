"""Backward-compatibility shim for sidebar UI component.

Re-exports sidebar and mobile navigation constants and components from
the authoritative canonical implementation in frontend.ui.sidebar.
"""

from __future__ import annotations

from frontend.ui.sidebar import (
    MOBILE_NAV_ITEMS,
    SIDEBAR_NAV_ITEMS,
    record_recent_search,
    render_mobile_navigation,
    render_sidebar,
)

__all__ = [
    "MOBILE_NAV_ITEMS",
    "SIDEBAR_NAV_ITEMS",
    "record_recent_search",
    "render_mobile_navigation",
    "render_sidebar",
]
