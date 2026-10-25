"""Research metrics and bibliometric visualization components."""

from __future__ import annotations

import collections
import html

import altair as alt
import pandas as pd
import streamlit as st

from agent.agent import ResearchAgentResult
from agent.overview import ResearchOverview


def render_research_metrics(result: ResearchAgentResult, overview: ResearchOverview) -> None:
    """Render scannable, data-grounded KPI cards for the research session."""
    ranked_papers = getattr(result, "ranked_papers", []) or []
    total_papers = len(ranked_papers)
    dedup_count = getattr(result, "total_deduplicated", 0) or total_papers
    raw_retrieved = getattr(result, "total_retrieved", 0)
    queries_count = (
        len(result.plan.queries)
        if getattr(result, "plan", None) and getattr(result.plan, "queries", None)
        else 0
    )

    # 1. Recommended papers
    rec_papers = [p for p in ranked_papers if getattr(p, "is_recommended", None) is True]
    rec_count = len(rec_papers)
    rec_subtext = (
        f"{round(rec_count / total_papers * 100)}% high-relevance consensus"
        if total_papers > 0
        else "Evaluated candidate papers"
    )

    # 2. Open Access coverage
    oa_papers = [
        p
        for p in ranked_papers
        if getattr(p, "is_open_access", None) is True
        or getattr(p, "access_status", "") == "open"
    ]
    oa_count = len(oa_papers)
    oa_subtext = (
        f"{round(oa_count / total_papers * 100)}% verified full-text access"
        if total_papers > 0
        else "Direct PDF availability"
    )

    # 3. Citation metrics
    citations = [
        p.citation_count
        for p in ranked_papers
        if getattr(p, "citation_count", None) is not None
    ]
    if citations:
        total_citations = sum(citations)
        avg_citations = total_citations / len(citations)
        cite_val = f"{total_citations:,}"
        cite_subtext = f"Avg {avg_citations:.1f} citations / paper"
    else:
        cite_val = "N/A"
        cite_subtext = "Citation data not reported"

    # 4. Evidence Confidence & Abstract Coverage
    conf_raw = getattr(overview.verdict, "confidence", "") or "UNRATED"
    conf_upper = conf_raw.upper()
    conf_class = (
        f"metric-confidence-{conf_raw.lower()}"
        if conf_raw.lower() in ("high", "medium", "low")
        else ""
    )
    abstract_count = getattr(overview, "papers_with_abstracts", 0)
    analyzed_count = getattr(overview, "total_papers_analyzed", total_papers)
    conf_subtext = f"{abstract_count}/{analyzed_count} abstracts verified"

    # Retrieved subtext
    retrieved_subtext = (
        f"{raw_retrieved} records from {queries_count} queries"
        if queries_count
        else f"{raw_retrieved} total records"
    )

    m_cols = st.columns(5)
    with m_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card metric-card-kpi">
                <span class="metric-label">Retrieved Papers</span>
                <span class="metric-value">{dedup_count}</span>
                <span class="metric-subtext">{html.escape(retrieved_subtext)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card metric-card-kpi">
                <span class="metric-label">Recommended Papers</span>
                <span class="metric-value">{rec_count}</span>
                <span class="metric-subtext">{html.escape(rec_subtext)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_cols[2]:
        st.markdown(
            f"""
            <div class="metric-card metric-card-kpi">
                <span class="metric-label">Open Access</span>
                <span class="metric-value">{oa_count}</span>
                <span class="metric-subtext">{html.escape(oa_subtext)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_cols[3]:
        st.markdown(
            f"""
            <div class="metric-card metric-card-kpi">
                <span class="metric-label">Total Citations</span>
                <span class="metric-value">{cite_val}</span>
                <span class="metric-subtext">{html.escape(cite_subtext)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_cols[4]:
        st.markdown(
            f"""
            <div class="metric-card metric-card-kpi">
                <span class="metric-label">Evidence Confidence</span>
                <span class="metric-value {conf_class}">{html.escape(conf_upper)}</span>
                <span class="metric-subtext">{html.escape(conf_subtext)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")


def render_overview_visualizations(result: ResearchAgentResult) -> None:
    """Render empirical distributions across publication timeline, venues, scores, and citations."""
    ranked_papers = getattr(result, "ranked_papers", []) or []
    if not ranked_papers:
        return

    st.markdown('<div class="overview-visualizations-section">', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            <div class="overview-viz-header">
                <div class="overview-viz-tag">EVIDENCE DISTRIBUTION &amp; BIBLIOMETRICS</div>
                <h4 class="overview-viz-title">Literature Landscape &amp; Analytics</h4>
                <p class="overview-viz-desc">
                    Empirical distributions across publication timeline, peer-reviewed venues,
                    normalized composite scores, and verified citation impact.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

        # ------------------ Column 1: Timeline & Venues ------------------
        with col1:
            # 1. Publication Timeline
            years = [
                p.year
                for p in ranked_papers
                if getattr(p, "year", None) is not None
                and isinstance(p.year, int)
                and p.year > 1900
            ]
            st.markdown(
                """
                <div class="overview-chart-header">
                    <div class="overview-chart-titles">
                        <div class="overview-chart-title">Publication Timeline</div>
                        <div class="overview-chart-subtitle">Distribution of candidate literature across publication years</div>
                    </div>
                    <span class="overview-chart-badge">Years</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if years:
                year_counts = collections.Counter(years)
                sorted_years = sorted(year_counts.keys())
                df_year = pd.DataFrame({
                    "Publication Year": [str(y) for y in sorted_years],
                    "Papers": [year_counts[y] for y in sorted_years],
                })
                chart_year = (
                    alt.Chart(df_year)
                    .mark_bar(
                        color="#6941C6",
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4,
                    )
                    .encode(
                        x=alt.X(
                            "Publication Year:O",
                            axis=alt.Axis(
                                title="Publication Year",
                                labelAngle=0,
                                labelColor="#534F67",
                                titleColor="#1E1B2E",
                                grid=False,
                            ),
                        ),
                        y=alt.Y(
                            "Papers:Q",
                            axis=alt.Axis(
                                title="Papers Published",
                                tickMinStep=1,
                                labelColor="#534F67",
                                titleColor="#1E1B2E",
                                grid=True,
                                gridColor="#E2E0EC",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Publication Year:O", title="Year"),
                            alt.Tooltip("Papers:Q", title="Paper Count"),
                        ],
                    )
                    .properties(height=230)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart_year, use_container_width=True)
            else:
                st.markdown(
                    '<div class="overview-chart-empty">Publication year metadata not available in retrieved records.</div>',
                    unsafe_allow_html=True,
                )

            # 2. Top Publishing Venues
            venues = [
                p.venue.strip()
                for p in ranked_papers
                if getattr(p, "venue", None)
                and isinstance(p.venue, str)
                and p.venue.strip()
            ]
            st.markdown(
                """
                <div class="overview-chart-header" style="margin-top: 1.25rem;">
                    <div class="overview-chart-titles">
                        <div class="overview-chart-title">Top Publishing Venues &amp; Outlets</div>
                        <div class="overview-chart-subtitle">Leading journals, conferences, and preprint repositories</div>
                    </div>
                    <span class="overview-chart-badge">Venues</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if venues:
                venue_counts = collections.Counter(venues).most_common(5)
                df_venues = pd.DataFrame({
                    "Venue": [v[0] for v in venue_counts],
                    "Papers": [v[1] for v in venue_counts],
                })
                chart_venue = (
                    alt.Chart(df_venues)
                    .mark_bar(
                        color="#7F56D9",
                        cornerRadiusTopRight=4,
                        cornerRadiusBottomRight=4,
                    )
                    .encode(
                        y=alt.Y(
                            "Venue:N",
                            sort="-x",
                            axis=alt.Axis(
                                title="",
                                labelColor="#1E1B2E",
                                labelFontSize=11,
                                labelLimit=160,
                                grid=False,
                            ),
                        ),
                        x=alt.X(
                            "Papers:Q",
                            axis=alt.Axis(
                                title="Paper Count",
                                tickMinStep=1,
                                labelColor="#534F67",
                                titleColor="#1E1B2E",
                                grid=True,
                                gridColor="#E2E0EC",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Venue:N", title="Venue / Outlet"),
                            alt.Tooltip("Papers:Q", title="Paper Count"),
                        ],
                    )
                    .properties(height=230)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart_venue, use_container_width=True)
            else:
                st.markdown(
                    '<div class="overview-chart-empty">Venue metadata not reported by source indexers for this candidate set.</div>',
                    unsafe_allow_html=True,
                )

        # ------------------ Column 2: Scores & Citations ------------------
        with col2:
            # 3. Quality Score Ranking Comparison
            scored_papers = [
                p
                for p in ranked_papers
                if getattr(p, "score", None) is not None
                and isinstance(p.score, (int, float))
            ]
            st.markdown(
                """
                <div class="overview-chart-header">
                    <div class="overview-chart-titles">
                        <div class="overview-chart-title">Candidate Quality Score Comparison</div>
                        <div class="overview-chart-subtitle">Objective multi-factor score distribution (0.00 – 1.00)</div>
                    </div>
                    <span class="overview-chart-badge">Scores</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if scored_papers:
                top_scored = scored_papers[:6]
                labels = [
                    f"#{idx + 1} "
                    + (p.title[:24] + "…" if len(p.title) > 25 else p.title)
                    for idx, p in enumerate(top_scored)
                ]
                statuses = [
                    "Recommended"
                    if getattr(p, "is_recommended", None) is True
                    else "Standard"
                    for p in top_scored
                ]
                scores = [round(float(p.score), 3) for p in top_scored]
                full_titles = [p.title for p in top_scored]

                df_scores = pd.DataFrame({
                    "Paper": labels,
                    "Score": scores,
                    "Recommendation": statuses,
                    "Full Title": full_titles,
                })
                chart_score = (
                    alt.Chart(df_scores)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        y=alt.Y(
                            "Paper:N",
                            sort=None,
                            axis=alt.Axis(
                                title="",
                                labelColor="#1E1B2E",
                                labelFontSize=11,
                                labelLimit=170,
                                grid=False,
                            ),
                        ),
                        x=alt.X(
                            "Score:Q",
                            scale=alt.Scale(domain=[0.0, 1.0]),
                            axis=alt.Axis(
                                title="Composite Quality Score",
                                labelColor="#534F67",
                                titleColor="#1E1B2E",
                                grid=True,
                                gridColor="#E2E0EC",
                            ),
                        ),
                        color=alt.Color(
                            "Recommendation:N",
                            scale=alt.Scale(
                                domain=["Recommended", "Standard"],
                                range=["#6941C6", "#9E77ED"],
                            ),
                            legend=alt.Legend(
                                title="Status",
                                orient="bottom",
                                labelColor="#534F67",
                                titleColor="#1E1B2E",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Full Title:N", title="Paper Title"),
                            alt.Tooltip(
                                "Score:Q", title="Quality Score", format=".2f"
                            ),
                            alt.Tooltip("Recommendation:N", title="Status"),
                        ],
                    )
                    .properties(height=230)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart_score, use_container_width=True)
            else:
                st.markdown(
                    '<div class="overview-chart-empty">Quality scores not yet computed for candidate papers.</div>',
                    unsafe_allow_html=True,
                )

            # 4. Verified Citations of Key Papers
            cited_papers = [
                p
                for p in ranked_papers
                if getattr(p, "citation_count", None) is not None
                and isinstance(p.citation_count, int)
                and p.citation_count > 0
            ]
            st.markdown(
                """
                <div class="overview-chart-header" style="margin-top: 1.25rem;">
                    <div class="overview-chart-titles">
                        <div class="overview-chart-title">Citation Impact of Key Papers</div>
                        <div class="overview-chart-subtitle">Verified citation counts reported by academic indexers</div>
                    </div>
                    <span class="overview-chart-badge">Citations</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if cited_papers:
                top_cited = sorted(
                    cited_papers,
                    key=lambda p: p.citation_count or 0,
                    reverse=True,
                )[:5]
                c_labels = [
                    f"#{idx + 1} "
                    + (p.title[:24] + "…" if len(p.title) > 25 else p.title)
                    for idx, p in enumerate(top_cited)
                ]
                c_counts = [p.citation_count or 0 for p in top_cited]
                c_titles = [p.title for p in top_cited]

                df_cites = pd.DataFrame({
                    "Paper": c_labels,
                    "Citations": c_counts,
                    "Full Title": c_titles,
                })
                chart_cite = (
                    alt.Chart(df_cites)
                    .mark_bar(
                        color="#53389E",
                        cornerRadiusTopRight=4,
                        cornerRadiusBottomRight=4,
                    )
                    .encode(
                        y=alt.Y(
                            "Paper:N",
                            sort="-x",
                            axis=alt.Axis(
                                title="",
                                labelColor="#1E1B2E",
                                labelFontSize=11,
                                labelLimit=170,
                                grid=False,
                            ),
                        ),
                        x=alt.X(
                            "Citations:Q",
                            axis=alt.Axis(
                                title="Verified Citations",
                                labelColor="#534F67",
                                titleColor="#1E1B2E",
                                grid=True,
                                gridColor="#E2E0EC",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("Full Title:N", title="Paper Title"),
                            alt.Tooltip(
                                "Citations:Q",
                                title="Verified Citations",
                                format=",",
                            ),
                        ],
                    )
                    .properties(height=230)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart_cite, use_container_width=True)
            else:
                st.markdown(
                    '<div class="overview-chart-empty">Verified citation counts not yet recorded for this candidate set.</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
