"""Diagnostics and score methodology rendering components."""

from __future__ import annotations

import html

import streamlit as st

from agent.agent import ResearchAgentResult
from mock_data import (
    SCORE_COMPONENT_LABELS,
    SCORE_COMPONENT_WEIGHTS,
)
from shared.models import Paper


def render_planned_queries(result: ResearchAgentResult) -> None:
    """Render planned search queries with intent, rationale, and diagnostics."""
    with st.expander(f"Planned Search Queries ({len(result.plan.queries)})"):
        for q_idx, q in enumerate(result.plan.queries, 1):
            st.markdown(f"{q_idx}. **[{q.intent.upper()}]** `{q.query}`")
            st.caption(f"Rationale: {q.rationale}")

        if result.diagnostics:
            st.markdown("---")
            st.markdown("**Retrieval Diagnostics:**")
            for diag in result.diagnostics:
                status_lower = diag.status.lower()
                if status_lower == "success":
                    badge = '<span class="diag-badge diag-badge-success">SUCCESS</span>'
                elif status_lower == "empty":
                    badge = '<span class="diag-badge diag-badge-empty">EMPTY (0 PAPERS)</span>'
                else:
                    badge = '<span class="diag-badge diag-badge-failed">FAILED</span>'

                err_text = f" — <span class='diag-error'>Error: {html.escape(diag.error_message)}</span>" if diag.error_message else ""
                st.markdown(
                    f"<div class='diag-item'>{badge} <code>{html.escape(diag.query)}</code>: "
                    f"<strong>{diag.paper_count}</strong> papers retrieved{err_text}</div>",
                    unsafe_allow_html=True,
                )


def render_score_analysis(ranked_papers: list[Paper]) -> None:
    """Render visual score analysis section explaining weights and distribution."""
    st.markdown('<div id="scores-section"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Score Analysis & Methodology")
        st.caption(
            "Papers are ranked using an objective multi-criteria model evaluating citation velocity, "
            "influential citations, venue prestige, publication recency, and peer review status. "
            "Scores provide a structured heuristic for literature prioritization, not an absolute measure of scientific validity."
        )

        cols = st.columns(len(SCORE_COMPONENT_WEIGHTS))
        for idx, (comp_key, weight) in enumerate(SCORE_COMPONENT_WEIGHTS.items()):
            label = SCORE_COMPONENT_LABELS[comp_key]
            with cols[idx]:
                st.markdown(
                    f'<div class="metric-card metric-card-compact">'
                    f'<span class="metric-label">{label}</span>'
                    f'<span class="metric-value">{int(weight * 100)}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.write("")

        if ranked_papers:
            st.markdown("**Top Ranked Literature Candidates:**")
            lines = [
                "| Rank | Paper Title | Score | Recommended | Citations | Year |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
            for idx, p in enumerate(ranked_papers[:10], 1):
                rec = "Yes" if p.is_recommended else "No"
                score_str = f"{p.score:.2f}" if p.score is not None else "N/A"
                title_short = (p.title[:60] + "...") if len(p.title) > 63 else p.title
                lines.append(
                    f"| #{idx} | {title_short} | {score_str} | {rec} | {p.citation_count or 0} | {p.year or 'N/A'} |"
                )
            st.markdown("\n".join(lines))
