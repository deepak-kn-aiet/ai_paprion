"""Research metrics rendering component."""

from __future__ import annotations

import streamlit as st

from agent.agent import ResearchAgentResult
from agent.overview import ResearchOverview


def render_research_metrics(result: ResearchAgentResult, overview: ResearchOverview) -> None:
    """Render compact, scannable research execution and evidence metrics."""
    m_cols = st.columns(5)
    with m_cols[0]:
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Queries Planned</span>'
            f'<span class="metric-value">{len(result.plan.queries)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with m_cols[1]:
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Retrieved Records</span>'
            f'<span class="metric-value">{result.total_retrieved}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with m_cols[2]:
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Deduplicated Works</span>'
            f'<span class="metric-value">{result.total_deduplicated}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with m_cols[3]:
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Analyzed Evidence</span>'
            f'<span class="metric-value">{overview.total_papers_analyzed}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with m_cols[4]:
        conf_str = (overview.verdict.confidence or "UNRATED").upper()
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Evidence Confidence</span>'
            f'<span class="metric-value">{conf_str}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.write("")
