"""Initial dashboard component rendered before user initiates an active research query."""

from __future__ import annotations

import html

import streamlit as st

EXAMPLE_TOPICS = [
    {
        "title": "Retrieval-Augmented Generation for Clinical Medicine",
        "query": "Retrieval-augmented generation for biomedical question answering",
        "domain": "Biomedical NLP",
        "description": "Retriever-generator architectures, clinical hallucination mitigation, and medical evidence grounding.",
    },
    {
        "title": "Mechanistic Interpretability in Foundation Models",
        "query": "Mechanistic interpretability in large language models",
        "domain": "Safety & Alignment",
        "description": "Circuit tracing, induction heads, linear representations, and steering vectors in transformer models.",
    },
    {
        "title": "Diffusion Models for De Novo Molecular Graph Generation",
        "query": "Diffusion models for molecular graph generation",
        "domain": "Drug Discovery",
        "description": "Equivariant 3D molecular diffusion, SE(3) graph architectures, and generative chemical space exploration.",
    },
    {
        "title": "Renewable Energy Forecasting & Grid Demand Optimization",
        "query": "Renewable energy forecasting and grid demand optimization",
        "domain": "Energy & Climate",
        "description": "Spatio-temporal graph neural networks, solar/wind power forecasting, and decarbonization microgrid dynamics.",
    },
]


def render_initial_dashboard() -> None:
    """Render the purposeful academic research dashboard empty-state."""
    st.markdown('<div class="initial-dashboard-wrapper">', unsafe_allow_html=True)

    # 1. Clear 3-Step Autonomous Workflow Cue (Discover -> Synthesize -> Evaluate)
    st.markdown(
        """
        <div class="workflow-cue-strip">
            <div class="workflow-cue-card">
                <div class="workflow-cue-header">
                    <span class="workflow-cue-step">01</span>
                    <span class="workflow-cue-icon">🔍</span>
                    <strong class="workflow-cue-name">Discover</strong>
                </div>
                <p class="workflow-cue-desc">Find canonical research papers across 250M+ scientific records with automated query decomposition.</p>
            </div>
            <div class="workflow-cue-card">
                <div class="workflow-cue-header">
                    <span class="workflow-cue-step">02</span>
                    <span class="workflow-cue-icon">🧠</span>
                    <strong class="workflow-cue-name">Synthesize</strong>
                </div>
                <p class="workflow-cue-desc">Summarize core themes, consensus findings, and open gaps with Gemini AI evidence grounding.</p>
            </div>
            <div class="workflow-cue-card">
                <div class="workflow-cue-header">
                    <span class="workflow-cue-step">03</span>
                    <span class="workflow-cue-icon">⚖️</span>
                    <strong class="workflow-cue-name">Evaluate</strong>
                </div>
                <p class="workflow-cue-desc">Assess paper quality, citation velocity, venue tier (CORE/SJR), and methodological recency decay.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Curated Example Research Topics
    st.markdown(
        """
        <div class="initial-section-header">
            <span class="initial-section-tag">CURATED RESEARCH INQUIRIES</span>
            <h3 class="initial-section-title">Explore Canonical Academic Benchmarks</h3>
            <p class="initial-section-desc">
                Select a research question below to populate the search objective, or enter your own custom query above:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards_html: list[str] = []
    for idx, item in enumerate(EXAMPLE_TOPICS):
        escaped_query = html.escape(item["query"]).replace("'", "\\'")
        card_markup = f"""
        <div class="topic-preset-card" role="button" tabindex="0"
             onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();this.click();}}"
             onclick="(function(){{
                try {{
                    var doc = (window.parent && window.parent.document) ? window.parent.document : document;
                    var inp = doc.querySelector('[data-testid=\\'stTextInput\\'] input');
                    if (inp) {{
                        var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                        nativeSetter.call(inp, '{escaped_query}');
                        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        inp.focus();
                    }}
                }} catch(e){{}}
             }})()">
            <div class="topic-preset-meta">
                <span class="topic-preset-badge">{html.escape(item['domain'])}</span>
                <span class="topic-preset-action">Click to populate ↗</span>
            </div>
            <div class="topic-preset-title">{html.escape(item['title'])}</div>
            <div class="topic-preset-desc">{html.escape(item['description'])}</div>
        </div>
        """
        cards_html.append(card_markup)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(cards_html[0] + cards_html[2], unsafe_allow_html=True)
    with col2:
        st.markdown(cards_html[1] + cards_html[3], unsafe_allow_html=True)

    st.markdown('<div class="initial-divider"></div>', unsafe_allow_html=True)

    # 3. Core Academic Foundations & Trust Strip
    st.markdown(
        """
        <div class="academic-trust-strip">
            <div class="trust-item">
                <span class="trust-icon">🔬</span>
                <div class="trust-text">
                    <strong>Real Literature Graph</strong>
                    <span>Live retrieval across 250M+ OpenAlex scientific records</span>
                </div>
            </div>
            <div class="trust-item">
                <span class="trust-icon">🔓</span>
                <div class="trust-text">
                    <strong>Open-Access Resolution</strong>
                    <span>Automated full-text PDF discovery via Unpaywall integration</span>
                </div>
            </div>
            <div class="trust-item">
                <span class="trust-icon">🎯</span>
                <div class="trust-text">
                    <strong>Zero-Hallucination Citations</strong>
                    <span>Every claim strictly mapped to canonical DOI or arXiv identifiers</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


PAGE_DESCRIPTIONS = {
    "Research Overview": "View executive research metrics, bibliometric distributions across publication timeline, top venues, and citations, narrative synthesis overview, and core thematic clusters.",
    "Paper Library": "Browse ranked academic papers, inspect metric badges, filter by Open Access, and explore complete score breakdowns.",
    "Score Analysis": "Analyze multi-factor scoring methodology, component weights (citations, venue, recency), and query diagnostics.",
    "Research Verdict": "Read the definitive evidence-grounded verdict, confidence rationale, key findings, and open research challenges.",
}

PAGE_FEATURES = {
    "Research Overview": [
        "Executive AI synthesis grounded in verified scientific literature",
        "Empirical bibliometrics across publication timeline, top venues, and citations",
        "Thematic clusters grouping related methodologies and breakthroughs",
        "Calibrated confidence assessment with abstract verification metrics",
    ],
    "Paper Library": [
        "Ranked candidate papers sorted by multi-criteria relevance scoring",
        "Open-access resolution with direct links to unpaywalled PDFs",
        "Dynamic filtering by venue, recommendation status, and keywords",
    ],
    "Score Analysis": [
        "Transparent component weighting (citations, venue tier, recency, peer review)",
        "Tabular ranking breakdown for all candidate literature",
        "Execution diagnostics for decomposed search queries",
    ],
    "Research Verdict": [
        "Definitive consensus takeaway and calibrated confidence score",
        "Key findings linked to canonical DOI/arXiv records",
        "Identified methodological limitations and open research challenges",
    ],
}


def render_empty_results_state(page_title: str) -> None:
    """Render a helpful empty state when a results page is accessed before running research."""
    escaped_title = html.escape(page_title)
    guidance = PAGE_DESCRIPTIONS.get(
        page_title,
        "Explore literature syntheses, quality evaluations, and academic evidence.",
    )
    features = PAGE_FEATURES.get(page_title, [])
    feature_items = "".join(f"<li>{html.escape(f)}</li>" for f in features)
    features_html = (
        f"<ul class='empty-results-feature-list'>{feature_items}</ul>"
        if feature_items
        else ""
    )

    st.markdown(
        f"""
        <div class="empty-results-box">
            <div class="empty-results-icon">🔬</div>
            <h3 class="empty-results-title">No Active Research for {escaped_title}</h3>
            <p class="empty-results-desc">
                No literature inquiry has been executed in the current session.
                Please start on the <strong>Research Search</strong> page to retrieve canonical literature,
                generate evidence syntheses, and evaluate candidate papers.
            </p>
            <div class="empty-results-hint">
                <strong>What you'll find here:</strong> {html.escape(guidance)}
                {features_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    btn_key = f"return_to_search_{page_title.lower().replace(' ', '_')}"
    if st.button("← Go to Research Search", key=btn_key, type="primary"):
        st.session_state["pending_navigation"] = "Research Search"
        st.rerun()
