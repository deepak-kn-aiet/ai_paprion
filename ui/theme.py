"""Theme configuration and CSS injection for AI Research Workspace."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

THEME_CSS_PATH = Path(__file__).resolve().parent / "styles.css"


def load_theme_css() -> str:
    """Load the design stylesheet from ui/styles.css."""
    try:
        return THEME_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def render_theme() -> None:
    """Inject custom stylesheet into Streamlit."""
    css = load_theme_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
