"""Diagnostics and score methodology rendering components."""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from agent.agent import ResearchAgentResult
from mock_data import (
    SCORE_COMPONENT_LABELS,
    SCORE_COMPONENT_WEIGHTS,
)
from shared.models import Paper

COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "citation_impact": "Normalized annual citation velocity relative to cohort maximum (max-scaling).",
    "influential_citation_ratio": "Direct methodology adoption ratio (influential citations / total citations).",
    "venue_quality": "Tiered prestige rating (Tier 1 = 1.0, Tier 2 = 0.85, Tier 3 = 0.70, Preprints = 0.40).",
    "recency": "Linear decay over a 10-year window from 1.0 (current year) to 0.0 (10+ years old).",
    "review_score": "Normalized review score (redistributed proportionally across remaining 4 when unindexed).",
}

RECOMMENDATION_THRESHOLD = 0.70


def render_planned_queries(result: ResearchAgentResult) -> None:
    """Render planned search queries with intent, rationale, and diagnostics."""
    queries_count = len(result.plan.queries) if getattr(result, "plan", None) and result.plan.queries else 0
    with st.expander(f"Planned Search Queries ({queries_count})"):
        if getattr(result, "plan", None) and result.plan.queries:
            for q_idx, q in enumerate(result.plan.queries, 1):
                st.markdown(f"{q_idx}. **[{q.intent.upper()}]** `{q.query}`")
                st.caption(f"Rationale: {q.rationale}")

        if getattr(result, "diagnostics", None):
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

                err_text = (
                    f" — <span class='diag-error'>Error: {html.escape(diag.error_message)}</span>"
                    if diag.error_message
                    else ""
                )
                st.markdown(
                    f"<div class='diag-item'>{badge} <code>{html.escape(diag.query)}</code>: "
                    f"<strong>{diag.paper_count}</strong> papers retrieved{err_text}</div>",
                    unsafe_allow_html=True,
                )


def _format_cell_score(breakdown: dict[str, float] | None, comp_key: str, paper: Paper) -> str:
    """Format component score distinguishing genuine zero from unindexed or missing data."""
    if not breakdown:
        return "Unreported"

    val: float | None = None
    if comp_key == "citation_impact":
        val = breakdown.get("citations_per_year") if "citations_per_year" in breakdown else breakdown.get("citation_impact")
    elif comp_key == "review_score":
        if getattr(paper, "review_score", None) is None:
            return "N/A (Redistributed)"
        val = breakdown.get("peer_review") if "peer_review" in breakdown else breakdown.get("review_score")
    else:
        val = breakdown.get(comp_key)

    if val is None:
        return "Unreported"

    if val == 0.0:
        return "0.00"
    return f"{val:.2f}"


def render_score_analysis(ranked_papers: list[Paper]) -> None:
    """Render comprehensive score analysis section with methodology, distributions, and breakdowns."""
    st.markdown('<div id="scores-section"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Score Analysis & Methodology")
        st.caption(
            "Papers are ranked using an objective multi-criteria model evaluating citation velocity, "
            "influential citations, venue prestige, publication recency, and peer review status. "
            "Scores provide a structured heuristic for literature prioritization, not an absolute measure of scientific validity."
        )

        # 1. Base Component Weights & Methodology Cards
        st.markdown(
            """
            <div class="score-methodology-header">
                <span class="score-methodology-tag">EVALUATION METHODOLOGY</span>
                <div class="score-methodology-title">Multi-Criteria Component Weights</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(len(SCORE_COMPONENT_WEIGHTS))
        for idx, (comp_key, weight) in enumerate(SCORE_COMPONENT_WEIGHTS.items()):
            label = SCORE_COMPONENT_LABELS[comp_key]
            desc = COMPONENT_DESCRIPTIONS.get(comp_key, "")
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="metric-card metric-card-compact score-component-card">
                        <span class="metric-label">{label}</span>
                        <span class="metric-value">{int(weight * 100)}%</span>
                        <span class="score-component-desc">{html.escape(desc)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")

        if not ranked_papers:
            st.markdown(
                """
                <div class="empty-filter-state">
                    <div class="empty-filter-icon">⚖️</div>
                    <div class="empty-filter-title">No Scored Papers Available</div>
                    <p class="empty-filter-desc">
                        No evaluated papers are available for score analysis in the current session.
                        Run a research inquiry on the Research Search page to score candidate literature.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        scored_papers = [p for p in ranked_papers if p.score is not None]
        if not scored_papers:
            st.info("Retrieved papers do not currently have multi-criteria scores computed.")
            return

        # 2. Score Summary Metrics Strip
        total_scored = len(scored_papers)
        rec_papers = [p for p in scored_papers if getattr(p, "is_recommended", False) or (p.score is not None and p.score >= RECOMMENDATION_THRESHOLD)]
        rec_count = len(rec_papers)
        cand_count = total_scored - rec_count
        scores_list = [p.score for p in scored_papers if p.score is not None]
        max_score = max(scores_list) if scores_list else 0.0
        avg_score = (sum(scores_list) / len(scores_list)) if scores_list else 0.0

        st.markdown(
            f"""
            <div class="score-summary-strip">
                <span class="score-summary-item">🎯 Scored Works: <strong>{total_scored}</strong></span>
                <span class="score-summary-divider">·</span>
                <span class="score-summary-item">🌟 Recommended (≥ 0.70): <strong>{rec_count}</strong></span>
                <span class="score-summary-divider">·</span>
                <span class="score-summary-item">⚖️ Candidate (&lt; 0.70): <strong>{cand_count}</strong></span>
                <span class="score-summary-divider">·</span>
                <span class="score-summary-item">🏆 Highest Score: <strong>{max_score:.2f}</strong></span>
                <span class="score-summary-divider">·</span>
                <span class="score-summary-item">📊 Average Score: <strong>{avg_score:.2f}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Visual Score Comparison Chart with Recommendation Threshold (0.70)
        st.markdown(
            """
            <div class="overview-chart-header" style="margin-top: 1rem;">
                <div class="overview-chart-titles">
                    <div class="overview-chart-title">Quality Score Distribution &amp; Recommendation Tiers</div>
                    <div class="overview-chart-subtitle">
                        Comparative ranking across candidate literature. The vertical dashed line marks the 0.70 recommendation threshold.
                    </div>
                </div>
                <span class="overview-chart-badge">Cutoff: 0.70</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top_candidates = scored_papers[:12]
        c_labels = [
            f"#{idx + 1} " + (p.title[:28] + "…" if len(p.title) > 29 else p.title)
            for idx, p in enumerate(top_candidates)
        ]
        c_scores = [round(float(p.score or 0.0), 3) for p in top_candidates]
        c_statuses = [
            "Recommended (Score ≥ 0.70)"
            if (p.score or 0.0) >= RECOMMENDATION_THRESHOLD
            else "Candidate (Score < 0.70)"
            for p in top_candidates
        ]
        c_titles = [p.title for p in top_candidates]
        c_cites = [str(p.citation_count or 0) for p in top_candidates]
        c_years = [str(p.year or "N/A") for p in top_candidates]

        df_chart = pd.DataFrame({
            "Paper": c_labels,
            "Score": c_scores,
            "Status": c_statuses,
            "Full Title": c_titles,
            "Citations": c_cites,
            "Year": c_years,
        })

        bars = (
            alt.Chart(df_chart)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y(
                    "Paper:N",
                    sort=None,
                    axis=alt.Axis(
                        title="",
                        labelColor="#1E1B2E",
                        labelFontSize=11,
                        labelLimit=220,
                        grid=False,
                    ),
                ),
                x=alt.X(
                    "Score:Q",
                    scale=alt.Scale(domain=[0.0, 1.0]),
                    axis=alt.Axis(
                        title="Composite Quality Score (0.00 – 1.00)",
                        labelColor="#534F67",
                        titleColor="#1E1B2E",
                        grid=True,
                        gridColor="#E2E0EC",
                    ),
                ),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Recommended (Score ≥ 0.70)", "Candidate (Score < 0.70)"],
                        range=["#6941C6", "#9E77ED"],
                    ),
                    legend=alt.Legend(
                        title="Recommendation Tier",
                        orient="bottom",
                        labelColor="#534F67",
                        titleColor="#1E1B2E",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Full Title:N", title="Paper Title"),
                    alt.Tooltip("Score:Q", title="Composite Score", format=".2f"),
                    alt.Tooltip("Status:N", title="Recommendation Tier"),
                    alt.Tooltip("Citations:N", title="Citations"),
                    alt.Tooltip("Year:N", title="Year"),
                ],
            )
        )

        rule = (
            alt.Chart(pd.DataFrame({"Threshold": [RECOMMENDATION_THRESHOLD]}))
            .mark_rule(color="#D92D20", strokeDash=[4, 4], strokeWidth=2)
            .encode(x="Threshold:Q")
        )

        score_chart = (
            (bars + rule)
            .properties(height=max(180, len(df_chart) * 28))
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(score_chart, use_container_width=True)

        # 4. Detailed Component Breakdown Table
        st.markdown("**Top Ranked Literature Candidates & Component Breakdown:**")
        lines = [
            "| Rank | Paper Title | Score | Tier | Citations/Yr (40%) | Influential (20%) | Venue (20%) | Recency (10%) | Peer Review (10%) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for idx, p in enumerate(scored_papers[:15], 1):
            is_rec = (p.score or 0.0) >= RECOMMENDATION_THRESHOLD or bool(p.is_recommended)
            tier_str = "**Recommended**" if is_rec else "Candidate"
            score_str = f"**{p.score:.2f}**" if p.score is not None else "N/A"
            title_short = (p.title[:55] + "...") if len(p.title) > 58 else p.title

            c_cit = _format_cell_score(p.score_breakdown, "citation_impact", p)
            c_inf = _format_cell_score(p.score_breakdown, "influential_citation_ratio", p)
            c_ven = _format_cell_score(p.score_breakdown, "venue_quality", p)
            c_rec = _format_cell_score(p.score_breakdown, "recency", p)
            c_rev = _format_cell_score(p.score_breakdown, "review_score", p)

            lines.append(
                f"| #{idx} | {title_short} | {score_str} | {tier_str} | {c_cit} | {c_inf} | {c_ven} | {c_rec} | {c_rev} |"
            )
        st.markdown("\n".join(lines))

        # 5. Interactive Paper Score Inspector
        st.write("")
        paper_options = [
            f"#{idx + 1} - {p.title}" for idx, p in enumerate(scored_papers[:20])
        ]
        selected_option = st.selectbox(
            "Inspect individual paper score breakdown",
            paper_options,
            index=0,
            key="score_inspect_select",
        )
        if selected_option:
            selected_idx = paper_options.index(selected_option)
            selected_paper = scored_papers[selected_idx]
            breakdown = selected_paper.score_breakdown or {}

            # Effective weight calculation (if review_score is missing, redistribute to 40/90, 20/90, 20/90, 10/90)
            has_review = getattr(selected_paper, "review_score", None) is not None
            is_rec = (selected_paper.score or 0.0) >= RECOMMENDATION_THRESHOLD

            st.markdown(
                f"""
                <div class="score-inspector-card">
                    <div class="score-inspector-header">
                        <div>
                            <span class="paper-rank-badge">Candidate #{selected_idx + 1}</span>
                            <div class="score-inspector-title">{html.escape(selected_paper.title)}</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="metric-label">Composite Score</span>
                            <span class="score-pill {'score-pill-recommended' if is_rec else 'score-pill-candidate'}">
                                {selected_paper.score:.2f} / 1.00
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            insp_cols = st.columns(len(SCORE_COMPONENT_WEIGHTS))
            for c_idx, (comp_key, base_weight) in enumerate(SCORE_COMPONENT_WEIGHTS.items()):
                label = SCORE_COMPONENT_LABELS[comp_key]
                if comp_key == "review_score" and not has_review:
                    eff_weight = 0.0
                    val_str = "N/A (Redistributed)"
                    val_float = None
                else:
                    eff_weight = (base_weight / 0.90) if not has_review else base_weight
                    raw_val = _format_cell_score(breakdown, comp_key, selected_paper)
                    val_str = raw_val
                    try:
                        val_float = float(raw_val)
                    except ValueError:
                        val_float = None

                with insp_cols[c_idx]:
                    st.markdown(
                        f"""
                        <div class="metric-card metric-card-compact">
                            <span class="metric-label">{html.escape(label)}</span>
                            <span class="metric-value" style="font-size: 1.15rem;">{html.escape(val_str)}</span>
                            <span class="score-component-desc">Weight: {int(eff_weight * 100)}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if val_float is not None:
                        st.progress(max(0.0, min(1.0, val_float)))
