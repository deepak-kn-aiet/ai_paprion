"""Hero banner and research report header rendering components."""

from __future__ import annotations

import html

import streamlit as st

from agent.agent import ResearchAgentResult
from frontend.ui.hero_shader import render_hero_shader


def render_team_banner() -> None:
    """Render a compact, centered CODE X DYNASTY team branding banner."""
    st.markdown(
        """
        <div class="team-branding-banner-wrapper">
            <div class="team-branding-banner">
                <div class="team-banner-eyebrow">
                    <span class="team-eyebrow-dot"></span>
                    <span>MEET THE TEAM</span>
                </div>
                <div class="team-banner-main">
                    <span class="team-banner-icon" aria-hidden="true">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="16 18 22 12 16 6"></polyline>
                            <polyline points="8 6 2 12 8 18"></polyline>
                        </svg>
                    </span>
                    <span class="team-banner-name">CODE X DYNASTY</span>
                </div>
                <div class="team-banner-tagline">Built with intelligence.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render main editorial hero header and description with WebGL ShaderBackground backdrop."""
    render_team_banner()
    render_hero_shader()
    st.markdown(
        """
        <div id="hero-shader-mount" class="hero-shader-mount" aria-hidden="true">
            <div class="hero-shader-frame">
                <canvas id="hero-shader-canvas"></canvas>
                <div class="hero-shader-overlay"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-eyebrow">ACADEMIC INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("AI Research Workspace")
    st.markdown(
        """
        <div class="hero-desc">
            Search, analyze, and synthesize academic literature with AI. Get evidence-grounded insights,
            key findings, and a structured research overview.
        </div>
        <div class="editorial-divider"></div>
        """,
        unsafe_allow_html=True,
    )


def render_report_header(topic: str, result: ResearchAgentResult) -> None:
    """Render executive research synthesis report header."""
    import config

    st.markdown('<div id="report-header-section"></div>', unsafe_allow_html=True)
    clean_topic = topic.strip() if topic else "Academic Research Topic"
    status_badge = "Cached Result" if getattr(result, "is_cached", False) else "Pipeline Complete"
    provider_label = "Groq" if config.LLM_PROVIDER == "groq" else "Gemini"
    st.markdown(
        f"""
        <div class="report-header-card">
            <div class="report-eyebrow-tag">ACADEMIC RESEARCH REPORT</div>
            <div class="report-topic-title">{html.escape(clean_topic)}</div>
            <div class="report-meta-row">
                <span class="report-badge-complete">{status_badge}</span>
                <span class="report-meta-text">{result.total_deduplicated} verified papers synthesized</span>
                <span class="report-meta-dot">·</span>
                <span class="report-meta-text">{len(result.plan.queries)} queries executed</span>
                <span class="report-meta-dot">·</span>
                <span class="report-meta-text">OpenAlex retrieval & {provider_label} AI synthesis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
