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

    # Sorting & filtering controls
    ctrl_col1, ctrl_col2 = st.columns([2, 2])
    with ctrl_col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Quality Score (High to Low)", "Citations (High to Low)", "Year (Newest First)"],
            index=0,
        )
    with ctrl_col2:
        filter_opt = st.selectbox(
            "Filter recommendations",
            ["All Retrieved Papers", "Recommended Only (Score >= 0.70)", "Open Access Only"],
            index=0,
        )

    filtered = list(ranked_papers)
    if filter_opt == "Recommended Only (Score >= 0.70)":
        filtered = [p for p in filtered if p.is_recommended]
    elif filter_opt == "Open Access Only":
        filtered = [p for p in filtered if p.is_open_access]

    if sort_by == "Citations (High to Low)":
        filtered.sort(key=lambda p: (p.citation_count or 0), reverse=True)
    elif sort_by == "Year (Newest First)":
        filtered.sort(key=lambda p: (p.year or 0), reverse=True)
    else:
        # Default: Quality Score descending
        filtered.sort(key=lambda p: (p.score if p.score is not None else -1.0), reverse=True)

    st.caption(f"Displaying {len(filtered)} papers after applying filters.")
    for idx, paper in enumerate(filtered, 1):
        render_paper_card(build_paper_card(paper, clean_topic, rank=idx))
