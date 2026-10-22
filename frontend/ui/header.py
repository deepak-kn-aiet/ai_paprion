"""Hero banner and research report header rendering components."""

from __future__ import annotations

import html

import streamlit as st

from agent.agent import ResearchAgentResult
from frontend.ui.hero_shader import render_hero_shader


def render_hero() -> None:
    """Render main editorial hero header and description with WebGL ShaderBackground backdrop."""
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
    st.markdown('<div class="hero-eyebrow">AI-POWERED LITERATURE DISCOVERY</div>', unsafe_allow_html=True)
    st.title("AI Research Workspace")
    st.markdown(
        """
        <div class="hero-desc">
            Discover relevant papers, synthesize evidence across canonical literature,
            and evaluate research quality with calibrated, multi-factor scoring.
        </div>
        <div class="editorial-divider"></div>
        """,
        unsafe_allow_html=True,
    )


def render_report_header(topic: str, result: ResearchAgentResult) -> None:
    """Render executive research synthesis report header."""
    st.markdown('<div id="report-header-section"></div>', unsafe_allow_html=True)
    clean_topic = topic.strip() if topic else "Academic Research Topic"
    st.markdown(
        f"""
        <div class="report-header-card">
            <div class="report-eyebrow-tag">ACADEMIC RESEARCH REPORT</div>
            <div class="report-topic-title">{html.escape(clean_topic)}</div>
            <div class="report-meta-row">
                <span class="report-badge-complete">Pipeline Complete</span>
                <span class="report-meta-text">{result.total_deduplicated} verified papers synthesized</span>
                <span class="report-meta-dot">·</span>
                <span class="report-meta-text">{len(result.plan.queries)} queries executed</span>
                <span class="report-meta-dot">·</span>
                <span class="report-meta-text">OpenAlex retrieval & Gemini AI synthesis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
