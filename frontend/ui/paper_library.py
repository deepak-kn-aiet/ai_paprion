"""Paper library component with sorting and filtering controls."""

from __future__ import annotations

import streamlit as st

from shared.models import Paper

from .formatters import build_paper_card
from .paper_card import render_paper_card


def render_paper_library(ranked_papers: list[Paper], clean_topic: str) -> None:
    """Render paper library with filtering and sorting controls."""
    st.markdown('<div id="library-section"></div>', unsafe_allow_html=True)
    st.subheader(f"Paper Library ({len(ranked_papers)} papers)")

    if not ranked_papers:
        st.info("No papers available in the library.")
        return

    # Literature overview summary strip
    total_count = len(ranked_papers)
    rec_count = sum(1 for p in ranked_papers if getattr(p, "is_recommended", False))
    oa_count = sum(
        1
        for p in ranked_papers
        if getattr(p, "is_open_access", False)
        or getattr(p, "access_status", "") == "open"
    )
    pdf_count = sum(1 for p in ranked_papers if getattr(p, "pdf_url", None))
    years: list[int] = [p.year for p in ranked_papers if p.year is not None]
    year_range_str = f"{min(years)}–{max(years)}" if years else "N/A"

    st.markdown(
        f"""
        <div class="library-summary-strip">
            <span class="library-summary-item">📚 Total Retrieved: <strong>{total_count}</strong></span>
            <span class="library-summary-divider">·</span>
            <span class="library-summary-item">🌟 Recommended: <strong>{rec_count}</strong></span>
            <span class="library-summary-divider">·</span>
            <span class="library-summary-item">🔓 Open Access: <strong>{oa_count}</strong></span>
            <span class="library-summary-divider">·</span>
            <span class="library-summary-item">📄 Direct PDFs: <strong>{pdf_count}</strong></span>
            <span class="library-summary-divider">·</span>
            <span class="library-summary-item">📅 Years: <strong>{year_range_str}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sorting & filtering controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1.2])
    with ctrl_col1:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Quality Score (High to Low)",
                "Citations (High to Low)",
                "Influential Citations (High to Low)",
                "Year (Newest First)",
                "Year (Oldest First)",
                "Title (A to Z)",
            ],
            index=0,
            key="lib_sort_select",
        )
    with ctrl_col2:
        filter_opt = st.selectbox(
            "Filter recommendations",
            [
                "All Retrieved Papers",
                "Recommended Only (Score >= 0.70)",
                "Open Access Only",
                "Direct PDF Available",
            ],
            index=0,
            key="lib_filter_select",
        )
    with ctrl_col3:
        venues: list[str] = sorted({p.venue for p in ranked_papers if p.venue and p.venue != "N/A"})
        venue_options = ["All Venues"] + venues if venues else ["All Venues"]
        venue_filter = st.selectbox(
            "Filter by venue",
            venue_options,
            index=0,
            key="lib_venue_select",
        )

    # Search within library input
    search_query = st.text_input(
        "Search within library",
        placeholder="Filter by title, author, venue, or keyword...",
        key="lib_filter_search_term",
        label_visibility="collapsed",
    )

    filtered = list(ranked_papers)
    if filter_opt == "Recommended Only (Score >= 0.70)":
        filtered = [p for p in filtered if getattr(p, "is_recommended", False)]
    elif filter_opt == "Open Access Only":
        filtered = [
            p
            for p in filtered
            if getattr(p, "is_open_access", False)
            or getattr(p, "access_status", "") == "open"
        ]
    elif filter_opt == "Direct PDF Available":
        filtered = [p for p in filtered if getattr(p, "pdf_url", None)]

    if venue_filter != "All Venues":
        filtered = [p for p in filtered if getattr(p, "venue", "") == venue_filter]

    if search_query and search_query.strip():
        q = search_query.strip().lower()
        filtered = [
            p
            for p in filtered
            if q in (getattr(p, "title", "") or "").lower()
            or q in (getattr(p, "abstract", "") or "").lower()
            or q in (getattr(p, "venue", "") or "").lower()
            or any(q in str(a).lower() for a in (getattr(p, "authors", []) or []))
        ]

    if sort_by == "Citations (High to Low)":
        filtered.sort(key=lambda p: (getattr(p, "citation_count", None) or 0), reverse=True)
    elif sort_by == "Influential Citations (High to Low)":
        filtered.sort(
            key=lambda p: (getattr(p, "influential_citation_count", None) or 0),
            reverse=True,
        )
    elif sort_by == "Year (Newest First)":
        filtered.sort(key=lambda p: (getattr(p, "year", None) or 0), reverse=True)
    elif sort_by == "Year (Oldest First)":
        filtered.sort(
            key=lambda p: (p.year if p.year is not None else 9999)
        )
    elif sort_by == "Title (A to Z)":
        filtered.sort(key=lambda p: (getattr(p, "title", "") or "").lower())
    else:
        # Default: Quality Score descending
        filtered.sort(
            key=lambda p: (p.score if p.score is not None else -1.0),
            reverse=True,
        )

    if not filtered:
        st.markdown(
            """
            <div class="empty-filter-state">
                <div class="empty-filter-icon">🔍</div>
                <div class="empty-filter-title">No papers match your filter criteria</div>
                <p class="empty-filter-desc">
                    Try clearing your search keyword, changing the recommendation filter, or selecting "All Venues".
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Reset Filters", key="reset_filters_btn"):
            for k in [
                "lib_filter_search_term",
                "lib_sort_select",
                "lib_filter_select",
                "lib_venue_select",
            ]:
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown('<div class="mobile-bottom-spacer"></div>', unsafe_allow_html=True)
        return

    active_filters = []
    if filter_opt != "All Retrieved Papers":
        active_filters.append(filter_opt)
    if venue_filter != "All Venues":
        active_filters.append(f"Venue: {venue_filter}")
    if search_query and search_query.strip():
        active_filters.append(f'"{search_query.strip()}"')

    if active_filters:
        filter_str = ", ".join(active_filters)
        st.caption(
            f"Displaying {len(filtered)} of {total_count} papers matching active filters ({filter_str})."
        )
    else:
        st.caption(f"Displaying {len(filtered)} papers.")

    for idx, paper in enumerate(filtered, 1):
        render_paper_card(build_paper_card(paper, clean_topic, rank=idx))

    st.markdown('<div class="mobile-bottom-spacer"></div>', unsafe_allow_html=True)
