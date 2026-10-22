"""Sidebar component with branding, navigation, and session search history."""

from __future__ import annotations

import html

import streamlit as st

SIDEBAR_NAV_ITEMS = [
    ("Research Search", "search-section"),
    ("Research Overview", "overview-section"),
    ("Paper Library", "library-section"),
    ("Score Analysis", "scores-section"),
    ("Research Verdict", "verdict-section"),
]


def record_recent_search(topic: str) -> None:
    """Record a completed search topic into session state (max 5 items)."""
    recent_searches = list(st.session_state.get("recent_searches", []))
    if topic not in recent_searches:
        recent_searches.insert(0, topic)
    st.session_state["recent_searches"] = recent_searches[:5]


def render_sidebar() -> str:
    """Render compact, professional sidebar with identity, nav, and real session search history."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand-box">
            <div class="sidebar-brand-badge">
                <span class="sidebar-brand-dot"></span>
                <span class="sidebar-brand-tag">RESEARCH INTELLIGENCE</span>
            </div>
            <div class="sidebar-title">
                <span class="sidebar-brand-icon">🔬</span>
                <span class="sidebar-brand-text">AI Paprion</span>
            </div>
            <div class="sidebar-desc">
                Autonomous literature discovery, multi-criteria paper scoring & grounded synthesis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    active_nav = st.sidebar.radio(
        "Navigation",
        [item[0] for item in SIDEBAR_NAV_ITEMS],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-label">Recent Searches</div>', unsafe_allow_html=True)
    recent_searches: list[str] = st.session_state.get("recent_searches", [])
    if recent_searches:
        for recent_topic in recent_searches:
            escaped_topic = html.escape(recent_topic)
            escaped_js = escaped_topic.replace("'", "\\'")
            st.sidebar.markdown(
                f"""
                <div class="recent-search" role="button" tabindex="0" title="{escaped_topic}"
                     onclick="(function(){{
                        try {{
                            var doc = (window.parent && window.parent.document) ? window.parent.document : document;
                            var inp = doc.querySelector('[data-testid=\\'stTextInput\\'] input');
                            if (inp) {{
                                var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                                nativeSetter.call(inp, '{escaped_js}');
                                inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                inp.focus();
                            }}
                        }} catch(e){{}}
                     }})()">
                    <span class="recent-search-icon">⏱</span>
                    <span class="recent-search-text">{escaped_topic}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.markdown(
            '<div class="sidebar-empty-state">No searches in current session.</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="sidebar-footer-box">
            <div class="sidebar-footer-tag">Powered by</div>
            <div class="sidebar-footer-text">OpenAlex Retrieval & Gemini AI Synthesis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return str(active_nav)
