"""Paper card, score breakdown, and metric badges rendering components."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from .formatters import SCORE_BREAKDOWN_NOTE


def render_metric_cards(card: dict[str, Any]) -> None:
    """Render scannable citation and score metrics for a paper card."""
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.markdown(
            '<div class="metric-card metric-card-compact">'
            '<span class="metric-label">Citations</span>'
            f'<span class="metric-value">{card["citations"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            '<div class="metric-card metric-card-compact">'
            '<span class="metric-label">Influential Citations</span>'
            f'<span class="metric-value">{card["influential_citations"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with metric_cols[2]:
        st.markdown(
            '<div class="metric-card metric-card-compact">'
            '<span class="metric-label">Overall Score (0-1)</span>'
            f'<span class="metric-value">{card["score"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        if card["score_value"] is not None:
            st.progress(card["score_value"])


def render_score_breakdown(card: dict[str, Any]) -> None:
    """Render score breakdown table inside an expander."""
    with st.expander(f"Score breakdown ({card['score']})"):
        st.caption(SCORE_BREAKDOWN_NOTE)
        lines = [
            "| Component | Score (0-1) | Weight | Weighted contribution |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for row in card["score_breakdown"]:
            if row["available"]:
                lines.append(
                    f"| {row['component']} | {row['score']:.2f} | "
                    f"{row['weight']:.2f} | {row['contribution']:.3f} |"
                )
            else:
                lines.append(
                    f"| {row['component']} | N/A | {row['weight']:.2f} | N/A |"
                )
        if card["weighted_total"] is None:
            total_text = "N/A"
        else:
            total_text = f"{card['weighted_total']:.3f}"
        lines.append(f"| **Weighted total** | | **1.00** | **{total_text}** |")
        st.markdown("\n".join(lines))


def render_paper_card(card: dict[str, Any]) -> None:
    """Render individual academic paper card with editorial layout and typography."""
    badge_class = "score-pill"
    if card.get("is_recommended") is True:
        badge_class += " score-pill-recommended"
    elif card.get("is_recommended") is False:
        badge_class += " score-pill-candidate"

    title_escaped = html.escape(str(card.get("title") or "Untitled Academic Work"))
    authors_escaped = html.escape(str(card.get("authors") or "Unknown Authors"))
    year_escaped = html.escape(str(card.get("year") or "N/A"))
    venue_escaped = html.escape(str(card.get("venue") or "Unspecified Venue"))

    tag_html = (
        '<div class="paper-meta-row">'
        f'<span class="tag tag-violet">Year: {year_escaped}</span>'
        f'<span class="tag">Venue: {venue_escaped}</span>'
    )
    if card.get("is_recommended") is True:
        tag_html += '<span class="tag tag-green">Recommended</span>'
    elif card.get("is_recommended") is False:
        tag_html += '<span class="tag tag-amber">Candidate</span>'
    if card.get("is_open_access") is True:
        tag_html += '<span class="tag tag-green">Open Access</span>'
    tag_html += "</div>"

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

    # Scannable metric cards (Citations, Influential Citations, Score)
    render_metric_cards(card)

    if card.get("summary"):
        summary_raw = str(card["summary"]).strip()
        if len(summary_raw) > 380:
            snippet_text = summary_raw[:260].rsplit(" ", 1)[0]
            remainder_text = summary_raw[len(snippet_text):].strip()
            st.markdown(
                f"""
                <div class="paper-abstract">
                    <strong>Abstract:</strong> {html.escape(snippet_text)}…
                    <details class="abstract-disclosure">
                        <summary class="abstract-toggle">Read full abstract ▾</summary>
                        <div class="abstract-full-text">… {html.escape(remainder_text)}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="paper-abstract"><strong>Abstract:</strong> {html.escape(summary_raw)}</div>',
                unsafe_allow_html=True,
            )

    render_score_breakdown(card)

    # Direct source and PDF links with styled anchor badges
    links = []
    if card.get("paper_link"):
        escaped_link = html.escape(str(card["paper_link"]))
        links.append(f'<a href="{escaped_link}" target="_blank" rel="noopener noreferrer" class="paper-action-link">Academic Record ↗</a>')
    if card.get("pdf_link"):
        escaped_pdf = html.escape(str(card["pdf_link"]))
        links.append(f'<a href="{escaped_pdf}" target="_blank" rel="noopener noreferrer" class="paper-action-link">Direct PDF ↗</a>')
    if card.get("doi"):
        escaped_doi = html.escape(str(card["doi"]))
        links.append(f'<a href="https://doi.org/{escaped_doi}" target="_blank" rel="noopener noreferrer" class="paper-action-link">DOI: {escaped_doi} ↗</a>')
    if links:
        st.markdown('<div class="paper-links">' + " ".join(links) + "</div>", unsafe_allow_html=True)
    else:
        st.caption("No external URLs or PDF links available for this record.")
    st.markdown("---")
