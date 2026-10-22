"""Executive overview, narrative synthesis, and research verdict components."""

from __future__ import annotations

import html
import re

import streamlit as st

from agent.agent import ResearchAgentResult
from agent.overview import ResearchOverview
from shared.models import Paper


def format_synthesis_citations(text: str) -> str:
    """Format narrative synthesis citations into styled editorial badge elements."""
    escaped = html.escape(text)
    pattern = r"\[([a-zA-Z0-9_\-\.]+:[^\]]+)\]"
    styled = re.sub(
        pattern,
        r'<span class="inline-citation">[\1]</span>',
        escaped,
    )
    paragraphs = [p.strip() for p in styled.split("\n\n") if p.strip()]
    if paragraphs:
        return "".join(f'<p class="synthesis-paragraph">{p}</p>' for p in paragraphs)
    return f'<p class="synthesis-paragraph">{styled}</p>'


def render_verdict_statement_box(overview: ResearchOverview) -> None:
    """Render executive verdict statement, confidence badge, and confidence rationale."""
    conf_str = (overview.verdict.confidence or "").strip()
    if conf_str:
        conf_class = f"verdict-confidence-{conf_str.lower()}"
        conf_badge = f'<span class="verdict-confidence-badge {conf_class}">CONFIDENCE: {html.escape(conf_str.upper())}</span>'
    else:
        conf_badge = ""

    rationale_html = ""
    if overview.verdict.confidence_rationale:
        rationale_html = (
            f'<div class="verdict-rationale">'
            f'<strong>Confidence Rationale:</strong> {html.escape(overview.verdict.confidence_rationale)}'
            f'</div>'
        )

    verdict_stmt = html.escape(overview.verdict.verdict_statement or "No definitive verdict statement available.")

    st.markdown(
        f"""
        <div class="verdict-box">
            <div class="verdict-header-row">
                <span class="verdict-box-title">EXECUTIVE VERDICT · {overview.total_papers_analyzed} PAPERS ANALYZED</span>
                {conf_badge}
            </div>
            <div class="verdict-box-statement">{verdict_stmt}</div>
            {rationale_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_overview(overview: ResearchOverview, result: ResearchAgentResult) -> None:
    """Render executive verdict, narrative synthesis, and research themes with editorial styling."""
    st.markdown('<div id="overview-section"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Executive Research Overview")
        render_verdict_statement_box(overview)

        # Narrative Synthesis
        if overview.synthesis:
            st.markdown("##### Evidence-Grounded Synthesis")
            st.markdown(
                f'<div class="synthesis-box">{format_synthesis_citations(overview.synthesis)}</div>',
                unsafe_allow_html=True,
            )

        # Thematic Clusters
        if overview.themes:
            with st.expander(f"Research Themes ({len(overview.themes)})", expanded=True):
                paper_lookup: dict[str, Paper] = {p.paper_id: p for p in result.ranked_papers}
                theme_cols = st.columns(2) if len(overview.themes) > 1 else [st.container()]
                for idx, theme in enumerate(overview.themes):
                    col = theme_cols[idx % len(theme_cols)]
                    with col:
                        resolved_items: list[str] = []
                        for pid in theme.paper_ids[:3]:
                            p = paper_lookup.get(pid)
                            if p:
                                venue_info = f" ({p.venue} {p.year})" if p.venue and p.year else (f" ({p.year})" if p.year else "")
                                title_snippet = p.title[:55] + ("..." if len(p.title) > 55 else "")
                                resolved_items.append(f"<li>{html.escape(title_snippet)}{venue_info}</li>")
                            else:
                                resolved_items.append(f"<li><code>{html.escape(pid)}</code></li>")

                        remaining = len(theme.paper_ids) - 3
                        if remaining > 0:
                            resolved_items.append(f"<li><em>+{remaining} additional papers</em></li>")

                        papers_html = f"<ul class='theme-paper-list'>{''.join(resolved_items)}</ul>" if resolved_items else ""

                        st.markdown(
                            f"""
                            <div class="theme-card">
                                <div class="theme-card-header">
                                    <span class="theme-card-num">Theme #{idx + 1}</span>
                                    <span class="theme-card-title">{html.escape(theme.name)}</span>
                                </div>
                                <div class="theme-card-summary">{html.escape(theme.summary)}</div>
                                <div class="theme-card-sub">Contributing papers ({len(theme.paper_ids)}):</div>
                                {papers_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


def render_verdict_section(overview: ResearchOverview) -> None:
    """Render findings, limitations, and verdict section in Light Editorial style."""
    st.markdown('<div id="verdict-section"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Research Verdict & Findings")
        render_verdict_statement_box(overview)

        # Key Findings
        if overview.verdict.key_findings:
            st.markdown(f"##### Key Findings ({len(overview.verdict.key_findings)})")
            for idx, finding in enumerate(overview.verdict.key_findings, 1):
                st.markdown(
                    f"""
                    <div class="finding-item">
                        <span class="finding-num">{idx:02d}</span>
                        <div class="finding-content">{html.escape(finding)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.write("")

        # Limitations & Open Research Gaps
        if overview.verdict.limitations_and_gaps:
            st.markdown(f"##### Limitations & Open Research Gaps ({len(overview.verdict.limitations_and_gaps)})")
            for gap in overview.verdict.limitations_and_gaps:
                st.markdown(
                    f"""
                    <div class="gap-box">
                        <span class="gap-tag">Challenge</span>
                        <div class="gap-content"><strong>Open Challenge:</strong> {html.escape(gap)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.write("")

        # Top Recommended Papers
        if overview.verdict.top_recommended_papers:
            rec_pills = " · ".join(f"<code>{html.escape(pid)}</code>" for pid in overview.verdict.top_recommended_papers)
            st.caption(f"**Top Recommended Candidate Literature:** {rec_pills}")
