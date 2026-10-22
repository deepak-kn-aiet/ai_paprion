"""Application footer component."""

from __future__ import annotations

import streamlit as st


def render_footer() -> None:
    """Render clean application footer."""
    st.markdown(
        """
        <div class="app-footer">
            <span class="footer-brand">AI Research Workspace</span>
            <span class="footer-sep">·</span>
            <span>Real OpenAlex Academic Retrieval & Gemini AI Synthesis</span>
            <span class="footer-sep">·</span>
            <span>Strict Referential Integrity</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
