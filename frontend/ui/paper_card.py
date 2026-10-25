"""Paper card, score breakdown, and metric badges rendering components."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from .formatters import SCORE_BREAKDOWN_NOTE


def render_metric_cards(card: dict[str, Any]) -> None:
    """Render scannable citation and score metrics for a paper card only when available."""
    metrics_to_show: list[tuple[str, str, str]] = []

    # Citations: only when available (not None, not 'N/A', and non-empty)
    citations_val = card.get("citations")
    if citations_val not in (None, "N/A", ""):
        metrics_to_show.append(("Citations", str(citations_val), "citations"))

    # Influential Citations: only when available
    influential_val = card.get("influential_citations")
    if influential_val not in (None, "N/A", ""):
        metrics_to_show.append(
            ("Influential Citations", str(influential_val), "influential")
        )

    # Overall Score: when available
    score_val = card.get("score")
    if score_val not in (None, "N/A", ""):
        metrics_to_show.append(("Overall Score (0-1)", str(score_val), "score"))

    if not metrics_to_show:
        return

    metric_cols = st.columns(len(metrics_to_show))
    for idx, (label, val, m_type) in enumerate(metrics_to_show):
        with metric_cols[idx]:
            st.markdown(
                f'<div class="metric-card metric-card-compact">'
                f'<span class="metric-label">{html.escape(label)}</span>'
                f'<span class="metric-value">{html.escape(val)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if m_type == "score" and card.get("score_value") is not None:
                st.progress(card["score_value"])


def render_score_breakdown(card: dict[str, Any]) -> None:
    """Render score breakdown table inside an expandable section."""
    score_label = card.get("score", "N/A")
    with st.expander(f"Score Breakdown ({score_label})", expanded=False):
        st.caption(SCORE_BREAKDOWN_NOTE)
        lines = [
            "| Component | Score (0-1) | Weight | Weighted contribution |",
            "| :--- | :--- | :--- | :--- |",
        ]
        breakdown_rows = card.get("score_breakdown") or []
        for row in breakdown_rows:
            if row.get("available"):
                score_num = row.get("score", 0.0)
                weight_num = row.get("weight", 0.0)
                contrib_num = row.get("contribution", 0.0)
                lines.append(
                    f"| {row['component']} | {score_num:.2f} | "
                    f"{weight_num:.2f} | {contrib_num:.3f} |"
                )
            else:
                weight_num = row.get("weight", 0.0)
                lines.append(
                    f"| {row['component']} | N/A | {weight_num:.2f} | N/A |"
                )
        if card.get("weighted_total") is None:
            total_text = "N/A"
        else:
            total_text = f"{card['weighted_total']:.3f}"
        lines.append(f"| **Weighted total** | | **1.00** | **{total_text}** |")
        st.markdown("\n".join(lines))


def render_paper_card(card: dict[str, Any]) -> None:
    """Render individual academic paper card with clean hierarchy, typography, and controls."""
    badge_class = "score-pill"
    if card.get("is_recommended") is True:
        badge_class += " score-pill-recommended"
    elif card.get("is_recommended") is False:
        badge_class += " score-pill-candidate"

    title_escaped = html.escape(str(card.get("title") or "Untitled Academic Work"))
    authors_raw = str(card.get("authors") or "").strip()
    authors_escaped = html.escape(authors_raw or "Unknown authors")

    year_raw = card.get("year")
    year_escaped = (
        html.escape(str(year_raw))
        if year_raw not in (None, "N/A", "")
        else None
    )

    venue_raw = str(card.get("venue") or "").strip()
    venue_escaped = (
        html.escape(venue_raw)
        if venue_raw not in (None, "N/A", "", "Unspecified Venue")
        else None
    )

    # Build meta tags row with graceful fallbacks
    tags: list[str] = []
    if year_escaped:
        tags.append(f'<span class="tag tag-violet">Year: {year_escaped}</span>')
    if venue_escaped:
        tags.append(f'<span class="tag">Venue: {venue_escaped}</span>')
    if card.get("is_recommended") is True:
        tags.append('<span class="tag tag-green">Recommended</span>')
    elif card.get("is_recommended") is False:
        tags.append('<span class="tag tag-amber">Candidate</span>')
    if card.get("is_open_access") is True:
        tags.append('<span class="tag tag-green">Open Access</span>')
    elif card.get("access_status") == "paywalled":
        tags.append('<span class="tag tag-neutral">Paywalled</span>')

    tag_html = f'<div class="paper-meta-row">{"".join(tags)}</div>' if tags else ""

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="paper-card">
                <div class="paper-card-top">
                    <div>
                        <div class="paper-rank-badge">Rank #{card.get('rank', 1)}</div>
                        <div class="paper-card-title">{title_escaped}</div>
                    </div>
                    <div>
                        <span class="{badge_class}">{card.get('score', 'N/A')}</span>
                    </div>
                </div>
                <div class="paper-authors">Authors: {authors_escaped}</div>
                {tag_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Scannable metric cards (Citations, Influential Citations, Score) - only when available
        render_metric_cards(card)

        # Abstract in an expandable section to prevent excessive card height
        summary_raw = str(card.get("summary") or "").strip()
        if summary_raw:
            with st.expander("Abstract", expanded=False):
                st.markdown(
                    f'<div class="paper-abstract"><strong>Abstract:</strong> {html.escape(summary_raw)}</div>',
                    unsafe_allow_html=True,
                )

        # Score breakdown expander
        render_score_breakdown(card)

        # Direct source, PDF, DOI, and arXiv links with clear action styling
        links: list[str] = []
        if card.get("paper_link"):
            escaped_link = html.escape(str(card["paper_link"]))
            links.append(
                f'<a href="{escaped_link}" target="_blank" rel="noopener noreferrer" class="paper-action-link paper-action-record">Academic Record ↗</a>'
            )
        if card.get("pdf_link"):
            escaped_pdf = html.escape(str(card["pdf_link"]))
            links.append(
                f'<a href="{escaped_pdf}" target="_blank" rel="noopener noreferrer" class="paper-action-link paper-action-pdf">Direct PDF ↗</a>'
            )
        if card.get("doi"):
            escaped_doi = html.escape(str(card["doi"]))
            links.append(
                f'<a href="https://doi.org/{escaped_doi}" target="_blank" rel="noopener noreferrer" class="paper-action-link paper-action-doi">DOI: {escaped_doi} ↗</a>'
            )
        if card.get("arxiv_id"):
            escaped_arxiv = html.escape(str(card["arxiv_id"]))
            links.append(
                f'<a href="https://arxiv.org/abs/{escaped_arxiv}" target="_blank" rel="noopener noreferrer" class="paper-action-link paper-action-arxiv">arXiv: {escaped_arxiv} ↗</a>'
            )

        if links:
            st.markdown(
                '<div class="paper-links">' + " ".join(links) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No external URLs or PDF links available for this record.")
