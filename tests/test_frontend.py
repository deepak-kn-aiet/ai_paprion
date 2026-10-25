"""Tests for the frontend modular UI components and application entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agent.overview import ResearchOverview, ResearchVerdict
from frontend.app import (
    build_paper_card,
    build_paper_overview,
    build_paper_verdict,
    build_score_breakdown,
    compute_weighted_score,
    format_authors,
    format_count,
    format_score,
    format_synthesis_citations,
    is_valid_component_score,
    is_valid_topic,
    load_theme_css,
    render_overview_page,
    render_verdict_page,
    run_agent_search,
    to_progress_value,
)
from frontend.ui.diagnostics import render_planned_queries, render_score_analysis
from frontend.ui.footer import render_footer
from frontend.ui.header import render_hero, render_report_header, render_team_banner
from frontend.ui.hero_shader import HERO_SHADER_HTML, render_hero_shader
from frontend.ui.metrics import (
    render_overview_visualizations,
    render_research_metrics,
)
from frontend.ui.paper_card import (
    render_metric_cards,
    render_paper_card,
    render_score_breakdown,
)
from frontend.ui.paper_library import render_paper_library
from frontend.ui.particles import SWARM_BACKGROUND_HTML, render_particle_background
from frontend.ui.sidebar import MOBILE_NAV_ITEMS, SIDEBAR_NAV_ITEMS, render_sidebar
from frontend.ui.theme import THEME_CSS_PATH, render_theme
from frontend.ui.tubes_cursor import TUBES_CURSOR_HTML, render_tubes_cursor
from frontend.ui.verdict import (
    extract_citation_tokens,
    format_authors_short,
    format_finding_citations,
    get_paper_external_url,
    render_executive_overview,
    render_verdict_section,
    render_verdict_statement_box,
)
from shared.models import Paper
from tests.test_app import _make_sample_agent_result

FRONTEND_APP_PATH = Path(__file__).resolve().parent.parent / "frontend" / "app.py"


def test_frontend_exports():
    """Verify all expected symbols are exported by frontend."""
    assert callable(render_team_banner)
    assert callable(render_hero)
    assert callable(render_report_header)
    assert callable(render_research_metrics)
    assert callable(render_overview_visualizations)
    assert callable(render_overview_page)
    assert callable(render_executive_overview)
    assert callable(render_verdict_section)
    assert callable(render_metric_cards)
    assert callable(render_score_breakdown)
    assert callable(render_paper_card)
    assert callable(render_paper_library)
    assert callable(render_particle_background)
    assert callable(render_planned_queries)
    assert callable(render_score_analysis)
    assert callable(render_sidebar)
    assert callable(render_footer)
    assert callable(render_theme)
    assert callable(load_theme_css)
    assert callable(run_agent_search)
    assert callable(is_valid_topic)
    assert callable(format_synthesis_citations)
    assert callable(build_paper_card)
    assert callable(build_paper_overview)
    assert callable(build_paper_verdict)
    assert callable(build_score_breakdown)
    assert callable(compute_weighted_score)
    assert callable(format_authors)
    assert callable(format_count)
    assert callable(format_score)
    assert callable(is_valid_component_score)
    assert callable(to_progress_value)
    assert callable(render_verdict_page)
    assert callable(render_verdict_statement_box)
    assert callable(extract_citation_tokens)
    assert callable(format_authors_short)
    assert callable(format_finding_citations)
    assert callable(get_paper_external_url)
    assert len(SIDEBAR_NAV_ITEMS) == 5
    assert len(MOBILE_NAV_ITEMS) == 5
    assert THEME_CSS_PATH.exists()


def test_frontend_theme_css_loads():
    css = load_theme_css()
    assert len(css) > 0
    assert "Academic Intelligence" in css or "body" in css or ".metric-card" in css


def test_frontend_app_initial_render():
    """Verify frontend/app.py renders initial clean UI with AppTest."""
    at = AppTest.from_file(FRONTEND_APP_PATH, default_timeout=30)
    at.run()
    assert len(at.title) >= 1
    assert at.title[0].value == "AI Research Workspace"
    assert len(at.text_input) == 1
    assert at.text_input[0].placeholder.startswith("e.g., Retrieval-augmented")
    assert len(at.button) >= 1
    assert at.button[0].label == "Run Research"
    assert len(at.error) == 0
    assert len(at.warning) == 0


def test_frontend_app_search_workflow():
    """Verify frontend/app.py executes search and renders all sections with mocked agent."""
    sample_result = _make_sample_agent_result("Applications of RAG")
    with patch("agent.agent.ResearchAgent") as mock_agent_cls:
        mock_instance = mock_agent_cls.return_value
        mock_instance.run.return_value = sample_result

        at = AppTest.from_file(FRONTEND_APP_PATH)
        at.run(timeout=30)
        at.text_input[0].input("Applications of RAG")
        at.button[0].click()
        at.run(timeout=30)

        assert len(at.success) >= 1
        assert "Pipeline Complete" in at.markdown[6].value or any("Pipeline Complete" in m.value for m in at.markdown)


def test_frontend_app_renders_hero_shader():
    """Verify that the hero backdrop renders correctly and capability strip is removed."""
    at = AppTest.from_file(FRONTEND_APP_PATH, default_timeout=30)
    at.run()
    # Ensure obsolete capability strip is absent
    assert not any('<div class="capability-strip">' in m.value for m in at.markdown)

    # Ensure hero backdrop mount is rendered
    shader_md = [m for m in at.markdown if '<div id="hero-shader-mount"' in m.value]
    assert len(shader_md) == 1
    val = shader_md[0].value
    assert "hero-tubes-canvas" in val or "hero-shader-canvas" in val
    assert "hero-shader-overlay" in val
    assert "```" not in val
    assert shader_md[0].allow_html is True

    # Ensure shader HTML script contains WebGL and accessibility safeguards
    assert "webgl" in HERO_SHADER_HTML
    assert "snoise" in HERO_SHADER_HTML
    assert "prefers-reduced-motion" in HERO_SHADER_HTML
    assert "visibilitychange" in HERO_SHADER_HTML
    assert "updateBounds" in HERO_SHADER_HTML
    assert callable(render_hero_shader)


def test_frontend_tubes_cursor():
    """Verify that TubesCursor component HTML contains 3D ribbon dynamics and safeguards."""
    assert "hero-tubes-canvas" in TUBES_CURSOR_HTML
    assert "TUBE_CONFIGS" in TUBES_CURSOR_HTML
    assert "prefers-reduced-motion" in TUBES_CURSOR_HTML
    assert "visibilitychange" in TUBES_CURSOR_HTML
    assert "updateBounds" in TUBES_CURSOR_HTML
    assert "requestAnimationFrame" in TUBES_CURSOR_HTML
    assert callable(render_tubes_cursor)


def test_frontend_particle_background():
    """Verify that particle background component HTML contains canvas and safety checks."""
    assert "swarm-cursor-canvas" in SWARM_BACKGROUND_HTML
    assert "prefers-reduced-motion" in SWARM_BACKGROUND_HTML
    assert "pointerEvents" in SWARM_BACKGROUND_HTML
    assert "requestAnimationFrame" in SWARM_BACKGROUND_HTML
    assert "doc.hidden" in SWARM_BACKGROUND_HTML
    assert callable(render_particle_background)


def test_build_paper_card_arxiv_and_access_status():
    """Verify build_paper_card preserves arxiv_id and access_status for Paper and dict objects."""
    from shared.models import Paper

    paper = Paper(
        paper_id="arxiv:2401.00001",
        title="Testing RAG",
        arxiv_id="2401.00001",
        access_status="open",
        is_open_access=True,
        is_recommended=True,
        score=0.85,
        year=2024,
        venue="NeurIPS",
    )
    card = build_paper_card(paper)
    assert card["arxiv_id"] == "2401.00001"
    assert card["access_status"] == "open"
    assert card["is_open_access"] is True

    dict_paper = {
        "paper_id": "arxiv:2401.00002",
        "title": "Dict Paper",
        "arxiv_id": "2401.00002",
        "access_status": "paywalled",
        "score": 0.60,
    }
    dict_card = build_paper_card(dict_paper)
    assert dict_card["arxiv_id"] == "2401.00002"
    assert dict_card["access_status"] == "paywalled"


def test_paper_library_summary_strip_and_filter_state():
    """Verify paper library renders summary strip, refined filters, and handles empty filter results."""
    from shared.models import Paper

    papers = [
        Paper(
            paper_id="p1",
            title="Attention Is All You Need",
            year=2017,
            venue="NeurIPS",
            citation_count=1000,
            score=0.95,
            is_recommended=True,
            is_open_access=True,
            pdf_url="http://example.com/pdf1",
        ),
        Paper(
            paper_id="p2",
            title="BERT: Pre-training",
            year=2019,
            venue="NAACL",
            citation_count=500,
            score=0.85,
            is_recommended=True,
            is_open_access=False,
        ),
    ]

    script = """
import streamlit as st
from frontend.ui.paper_library import render_paper_library

render_paper_library(st.session_state["test_papers"], "deep learning")
"""
    at = AppTest.from_string(script)
    at.session_state["test_papers"] = papers
    at.run()

    # Summary strip checks
    assert any("Total Retrieved: <strong>2</strong>" in m.value for m in at.markdown)
    assert any("Recommended: <strong>2</strong>" in m.value for m in at.markdown)
    assert any("Open Access: <strong>1</strong>" in m.value for m in at.markdown)
    assert any("Direct PDFs: <strong>1</strong>" in m.value for m in at.markdown)
    assert any("2017–2019" in m.value for m in at.markdown)

    # Filter controls checks
    assert len(at.selectbox) == 3
    assert at.selectbox[0].label == "Sort by"
    assert at.selectbox[1].label == "Filter recommendations"
    assert at.selectbox[2].label == "Filter by venue"
    assert len(at.text_input) == 1

    # Filter by search keyword matching nothing
    at.text_input[0].input("nonexistent quantum keyword").run()
    assert any("No papers match your filter criteria" in m.value for m in at.markdown)
    assert any("Reset Filters" in b.label for b in at.button)


def test_empty_results_state_hint():
    """Verify render_empty_results_state includes contextual feature guidance."""
    script = """
import streamlit as st
from frontend.ui.dashboard import render_empty_results_state

render_empty_results_state("Paper Library")
"""
    at = AppTest.from_string(script)
    at.run()
    assert any("No Active Research for Paper Library" in m.value for m in at.markdown)
    assert any("What you'll find here:" in m.value for m in at.markdown)


def test_team_branding_banner():
    """Verify CODE X DYNASTY team branding banner is rendered with expected content and markup."""
    script = """
import streamlit as st
from frontend.ui.header import render_team_banner

render_team_banner()
"""
    at = AppTest.from_string(script)
    at.run()
    assert len(at.markdown) >= 1
    val = at.markdown[0].value
    assert "team-branding-banner-wrapper" in val
    assert "team-branding-banner" in val
    assert "MEET THE TEAM" in val
    assert "CODE X DYNASTY" in val
    assert "Built with intelligence." in val
    assert "<svg" in val
    assert at.markdown[0].allow_html is True


def test_research_metrics_rendering():
    """Verify render_research_metrics computes accurate KPIs from real session data."""
    sample_result = _make_sample_agent_result("AI in Drug Discovery")
    script = """
import streamlit as st
from frontend.ui.metrics import render_research_metrics

render_research_metrics(st.session_state["res"], st.session_state["res"].overview)
"""
    at = AppTest.from_string(script)
    at.session_state["res"] = sample_result
    at.run()

    # Markdown elements should contain KPI labels
    combined = " ".join(m.value for m in at.markdown)
    assert "Retrieved Papers" in combined
    assert "Recommended Papers" in combined
    assert "Open Access" in combined
    assert "Total Citations" in combined
    assert "Evidence Confidence" in combined
    # 2 deduplicated, 1 recommended, 1 open access, 150+60=210 citations
    assert "210" in combined
    assert "HIGH" in combined
    assert "2/2 abstracts verified" in combined


def test_overview_visualizations_rendering():
    """Verify render_overview_visualizations renders bibliometric charts from real data."""
    sample_result = _make_sample_agent_result("AI in Drug Discovery")
    script = """
import streamlit as st
from frontend.ui.metrics import render_overview_visualizations

render_overview_visualizations(st.session_state["res"])
"""
    at = AppTest.from_string(script)
    at.session_state["res"] = sample_result
    at.run()

    combined = " ".join(m.value for m in at.markdown)
    assert "EVIDENCE DISTRIBUTION &amp; BIBLIOMETRICS" in combined or "EVIDENCE DISTRIBUTION" in combined
    assert "Literature Landscape &amp; Analytics" in combined or "Literature Landscape" in combined
    assert "Publication Timeline" in combined
    assert "Top Publishing Venues" in combined
    assert "Candidate Quality Score Comparison" in combined
    assert "Citation Impact of Key Papers" in combined
    # Verify Altair charts rendered without exception
    assert len(at.error) == 0
    assert len(at.get("arrow_vega_lite_chart")) >= 3


def test_overview_visualizations_empty_fallback():
    """Verify render_overview_visualizations handles empty ranked papers safely without error."""
    sample_result = _make_sample_agent_result("AI in Drug Discovery")
    sample_result.ranked_papers = []
    script = """
import streamlit as st
from frontend.ui.metrics import render_overview_visualizations

render_overview_visualizations(st.session_state["res"])
"""
    at = AppTest.from_string(script)
    at.session_state["res"] = sample_result
    at.run()
    assert len(at.error) == 0


def test_empty_results_state_research_overview():
    """Verify render_empty_results_state for Research Overview includes preview feature list."""
    script = """
import streamlit as st
from frontend.ui.dashboard import render_empty_results_state

render_empty_results_state("Research Overview")
"""
    at = AppTest.from_string(script)
    at.run()
    assert any("No Active Research for Research Overview" in m.value for m in at.markdown)
    combined = " ".join(m.value for m in at.markdown)
    assert "Executive AI synthesis" in combined
    assert "bibliometrics" in combined
    assert len(at.button) >= 1
    assert at.button[0].label == "← Go to Research Search"


def test_render_paper_card_conditional_metrics_and_hierarchy():
    """Verify render_paper_card only displays citation metric cards when data is available."""
    from frontend.ui.formatters import build_paper_card
    from shared.models import Paper

    # Paper WITH citations
    p_with_cites = Paper(
        paper_id="p-cites-1",
        title="Attention Is All You Need",
        citation_count=5000,
        influential_citation_count=1200,
        score=0.95,
        is_recommended=True,
    )
    card_with_cites = build_paper_card(p_with_cites, rank=1)

    script = """
import streamlit as st
from frontend.ui.paper_card import render_paper_card

render_paper_card(st.session_state["card"])
"""
    at = AppTest.from_string(script)
    at.session_state["card"] = card_with_cites
    at.run()

    combined = " ".join(m.value for m in at.markdown)
    assert "Citations" in combined
    assert "5,000" in combined
    assert "Influential Citations" in combined
    assert "1,200" in combined
    assert "Rank #1" in combined

    # Paper WITHOUT citations
    p_no_cites = Paper(
        paper_id="p-no-cites-2",
        title="Recent Preprint Without Citations",
        citation_count=None,
        influential_citation_count=None,
        score=0.80,
        is_recommended=True,
    )
    card_no_cites = build_paper_card(p_no_cites, rank=2)

    at2 = AppTest.from_string(script)
    at2.session_state["card"] = card_no_cites
    at2.run()

    combined2 = " ".join(m.value for m in at2.markdown)
    assert "Recent Preprint Without Citations" in combined2
    assert "Overall Score" in combined2
    # Verify Citations metric card was omitted, never fabricated as "N/A"
    assert "Influential Citations" not in combined2
    assert len(at2.error) == 0


def test_render_paper_card_missing_metadata_fallbacks():
    """Verify render_paper_card displays missing metadata gracefully without errors."""
    from frontend.ui.formatters import build_paper_card
    from shared.models import Paper

    # Paper with completely missing metadata
    sparse_paper = Paper(
        paper_id="p-sparse-3",
        title="Sparse Metadata Paper",
        authors=[],
        year=None,
        venue=None,
        score=None,
        abstract=None,
        url=None,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
    )
    sparse_card = build_paper_card(sparse_paper, rank=3)

    script = """
import streamlit as st
from frontend.ui.paper_card import render_paper_card

render_paper_card(st.session_state["card"])
"""
    at = AppTest.from_string(script)
    at.session_state["card"] = sparse_card
    at.run()

    combined = " ".join(m.value for m in at.markdown)
    assert "Sparse Metadata Paper" in combined
    assert "Authors: Unknown authors" in combined
    assert "Rank #3" in combined
    # Captions should indicate no external links
    assert any("No external URLs or PDF links available" in c.value for c in at.caption)
    assert len(at.error) == 0


def test_paper_library_sorting_controls():
    """Verify Paper Library sorting works across citation counts, years, and titles."""
    from shared.models import Paper

    papers = [
        Paper(
            paper_id="p-a",
            title="Alpha Paper",
            year=2020,
            citation_count=100,
            score=0.75,
        ),
        Paper(
            paper_id="p-b",
            title="Beta Paper",
            year=2024,
            citation_count=10,
            score=0.90,
        ),
        Paper(
            paper_id="p-g",
            title="Gamma Paper",
            year=2018,
            citation_count=500,
            score=0.60,
        ),
    ]

    script = """
import streamlit as st
from frontend.ui.paper_library import render_paper_library

render_paper_library(st.session_state["papers"], "testing topic")
"""
    at = AppTest.from_string(script)
    at.session_state["papers"] = papers
    at.run()

    # Default sort by Quality Score: Beta (0.90) > Alpha (0.75) > Gamma (0.60)
    titles = [m.value for m in at.markdown if "paper-card-title" in m.value]
    assert len(titles) == 3
    assert "Beta Paper" in titles[0]
    assert "Alpha Paper" in titles[1]
    assert "Gamma Paper" in titles[2]

    # Change sort to Citations (High to Low): Gamma (500) > Alpha (100) > Beta (10)
    at.selectbox[0].select("Citations (High to Low)").run()
    titles_cites = [m.value for m in at.markdown if "paper-card-title" in m.value]
    assert "Gamma Paper" in titles_cites[0]
    assert "Alpha Paper" in titles_cites[1]
    assert "Beta Paper" in titles_cites[2]


def test_render_score_analysis_rendering_and_threshold():
    """Verify render_score_analysis renders methodology, summary strip, chart, and table."""
    sample_result = _make_sample_agent_result("Evaluation of RAG")

    script = """
import streamlit as st
from frontend.ui.diagnostics import render_score_analysis

render_score_analysis(st.session_state["papers"])
"""
    at = AppTest.from_string(script)
    at.session_state["papers"] = sample_result.ranked_papers
    at.run()

    assert any("Score Analysis & Methodology" in s.value for s in at.subheader)
    combined = " ".join(m.value for m in at.markdown)
    assert "Multi-Criteria Component Weights" in combined
    assert "Scored Works: <strong>2</strong>" in combined
    assert "Recommended (≥ 0.70): <strong>1</strong>" in combined
    assert "Candidate (&lt; 0.70): <strong>1</strong>" in combined
    assert "Cutoff: 0.70" in combined
    assert "40%" in combined
    assert "20%" in combined
    assert "10%" in combined
    # Chart should render without errors
    assert len(at.error) == 0
    assert len(at.get("arrow_vega_lite_chart")) >= 1
    # Check inspector selectbox
    assert len(at.selectbox) >= 1
    assert at.selectbox[0].label == "Inspect individual paper score breakdown"


def test_render_score_analysis_missing_values_distinction():
    """Verify missing/unindexed peer-review scores are distinguished from genuine 0.0."""
    from shared.models import Paper

    papers = [
        Paper(
            paper_id="p-unindexed-rev",
            title="Preprint with Unindexed Review",
            score=0.75,
            review_score=None,  # unindexed review
            score_breakdown={
                "citations_per_year": 0.80,
                "influential_citation_ratio": 0.50,
                "venue_quality": 0.00,  # genuine 0.0 venue score
                "recency": 0.90,
                "peer_review": 0.00,  # 0 in breakdown because redistributed
            },
            is_recommended=True,
        )
    ]

    script = """
import streamlit as st
from frontend.ui.diagnostics import render_score_analysis

render_score_analysis(st.session_state["papers"])
"""
    at = AppTest.from_string(script)
    at.session_state["papers"] = papers
    at.run()

    combined = " ".join(m.value for m in at.markdown)
    # Review score unindexed should display N/A (Redistributed)
    assert "N/A (Redistributed)" in combined
    # Genuine zero venue score should display 0.00
    assert "0.00" in combined
    assert len(at.error) == 0


def test_render_score_analysis_empty_state():
    """Verify render_score_analysis handles empty paper lists safely with clean empty state."""
    script = """
import streamlit as st
from frontend.ui.diagnostics import render_score_analysis

render_score_analysis([])
"""
    at = AppTest.from_string(script)
    at.run()

    combined = " ".join(m.value for m in at.markdown)
    assert "No Scored Papers Available" in combined
    assert len(at.error) == 0


def test_render_verdict_section_complete_data():
    """Verify render_verdict_section displays executive conclusion, findings with literature grounding, gaps, and recommendations."""
    overview = ResearchOverview(
        topic="Retrieval-Augmented Generation Accuracy",
        total_papers_analyzed=12,
        papers_with_abstracts=10,
        verdict=ResearchVerdict(
            verdict_statement="RAG significantly mitigates hallucination with measurable accuracy gains across academic benchmarks.",
            confidence="high",
            confidence_rationale="Broad empirical consensus across 12 candidate papers with complete abstracts.",
            key_findings=[
                "External context grounding reduces hallucination rates substantially [doi:10.1234/test.rag1].",
                "Dense vector embeddings outperform sparse BM25 indexing on multi-hop questions [arxiv:2401.00001].",
            ],
            limitations_and_gaps=[
                "High latency in multi-stage cross-encoder re-ranking pipelines.",
                "Limited evaluation on non-English academic domains [doi:10.1234/test.rag1].",
            ],
            top_recommended_papers=["doi:10.1234/test.rag1", "arxiv:2401.00001"],
        ),
        synthesis="Retrieval-augmented approaches bridge parametric models with dynamic knowledge stores [doi:10.1234/test.rag1].",
        timings={"evidence_packet": 0.45, "synthesis": 1.25, "citation_validation": 0.08},
    )

    papers = [
        Paper(
            paper_id="doi:10.1234/test.rag1",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP",
            authors=["Lewis", "Yadav", "Deng"],
            year=2020,
            venue="NeurIPS",
            citation_count=4200,
            doi="10.1234/test.rag1",
            url="https://doi.org/10.1234/test.rag1",
            score=0.92,
            is_recommended=True,
        ),
        Paper(
            paper_id="arxiv:2401.00001",
            title="Dense Multi-Hop Retrieval Architectures",
            authors=["Smith", "Johnson"],
            year=2024,
            venue="arXiv preprint",
            citation_count=180,
            url="https://arxiv.org/abs/2401.00001",
            score=0.86,
            is_recommended=True,
        ),
    ]

    script = """
import streamlit as st
from frontend.ui.verdict import render_verdict_section

render_verdict_section(st.session_state["overview"], papers=st.session_state["papers"])
"""
    at = AppTest.from_string(script)
    at.session_state["overview"] = overview
    at.session_state["papers"] = papers
    at.run()

    # Subheaders check
    subheaders = [s.value for s in at.subheader]
    assert "Research Verdict & Findings" in subheaders

    combined_md = " ".join(m.value for m in at.markdown)
    # Executive conclusion elements
    assert "CONFIDENCE: HIGH" in combined_md
    assert "RESEARCH QUESTION" in combined_md
    assert "12 Papers Analyzed (10 with Abstracts)" in combined_md
    assert "RAG significantly mitigates hallucination" in combined_md
    assert "Confidence Justification:" in combined_md
    assert "Validation: 0.08s" in combined_md

    # Key findings & literature grounding
    assert "Finding 01" in combined_md
    assert "Finding 02" in combined_md
    assert "Supporting Literature Grounding" in combined_md
    assert "Retrieval-Augmented Generation for Knowledge-Intensive NLP" in combined_md
    assert "Lewis et al." in combined_md
    assert "NeurIPS (2020)" in combined_md
    assert "4,200 citations" in combined_md

    # Limitations & open challenges
    assert "Open Challenge" in combined_md
    assert "High latency in multi-stage cross-encoder re-ranking pipelines." in combined_md

    # Top recommended literature
    assert "Top Recommended Literature (2)" in combined_md
    assert "★ Recommended" in combined_md
    assert "Score: 0.92" in combined_md
    assert "Score: 0.86" in combined_md
    assert "DOI ↗" in combined_md
    assert "arXiv ↗" in combined_md

    # No uncaught exceptions
    assert len(at.error) == 0


def test_render_verdict_section_missing_optional_fields():
    """Verify render_verdict_section gracefully handles empty findings, empty gaps, and minimal data."""
    overview = ResearchOverview(
        topic="Sparse Topic",
        total_papers_analyzed=2,
        papers_with_abstracts=0,
        verdict=ResearchVerdict(
            verdict_statement="Inconclusive scientific consensus due to sparse literature.",
            confidence="low",
            confidence_rationale="",
            key_findings=[],
            limitations_and_gaps=[],
            top_recommended_papers=[],
        ),
        synthesis="Minimal synthesis narrative.",
        timings={},
    )

    script = """
import streamlit as st
from frontend.ui.verdict import render_verdict_section

render_verdict_section(st.session_state["overview"], papers=[])
"""
    at = AppTest.from_string(script)
    at.session_state["overview"] = overview
    at.run()

    combined_md = " ".join(m.value for m in at.markdown)
    assert "CONFIDENCE: LOW" in combined_md
    assert "2 Papers Analyzed" in combined_md
    assert "Inconclusive scientific consensus" in combined_md
    # Empty notices
    assert "No verified key factual findings were extracted" in combined_md
    assert "No critical methodological limitations or research gaps reported" in combined_md

    assert len(at.error) == 0


def test_render_verdict_section_unindexed_citation_fallback():
    """Verify render_verdict_section cleanly handles citations to unindexed papers without inventing data."""
    overview = ResearchOverview(
        topic="Unindexed Citations Test",
        total_papers_analyzed=1,
        papers_with_abstracts=1,
        verdict=ResearchVerdict(
            verdict_statement="Preliminary finding with external citation.",
            confidence="medium",
            confidence_rationale="Evidence based on isolated preprint reference.",
            key_findings=[
                "Evidence suggests linear scaling [doi:10.9999/unindexed.study].",
            ],
            limitations_and_gaps=[],
            top_recommended_papers=["doi:10.9999/unindexed.study"],
        ),
        synthesis="Synthesis citing [doi:10.9999/unindexed.study].",
        timings={},
    )

    script = """
import streamlit as st
from frontend.ui.verdict import render_verdict_section

# No papers in paper_lookup - testing fallback behavior
render_verdict_section(st.session_state["overview"], papers=[])
"""
    at = AppTest.from_string(script)
    at.session_state["overview"] = overview
    at.run()

    combined_md = " ".join(m.value for m in at.markdown)
    assert "CONFIDENCE: MEDIUM" in combined_md
    assert "[doi:10.9999/unindexed.study]" in combined_md
    assert "https://doi.org/10.9999/unindexed.study" in combined_md
    assert "Cited References (1)" in combined_md
    assert "★ Recommended Candidate" in combined_md
    assert "View Record ↗" in combined_md

    assert len(at.error) == 0


def test_render_verdict_page_empty_state():
    """Verify render_verdict_page renders clean empty state when no research session exists."""
    script = """
import streamlit as st
from frontend.app import render_verdict_page

render_verdict_page(None)
"""
    at = AppTest.from_string(script)
    at.run()

    combined_md = " ".join(m.value for m in at.markdown)
    assert "Research Verdict" in combined_md
    assert len(at.error) == 0


def test_verdict_helper_functions():
    """Unit test verdict citation and URL extraction helpers."""
    # extract_citation_tokens
    text = "RAG combines retrieval [doi:10.1000/1] with generator models [arxiv:2301.00000] and duplicate [doi:10.1000/1]."
    tokens = extract_citation_tokens(text)
    assert tokens == ["doi:10.1000/1", "arxiv:2301.00000"]
    assert extract_citation_tokens("") == []
    assert extract_citation_tokens("no citations here") == []

    # format_authors_short
    assert format_authors_short([]) == "Unknown Authors"
    assert format_authors_short(["Vaswani"]) == "Vaswani"
    assert format_authors_short(["Devlin", "Chang"]) == "Devlin &amp; Chang"
    assert format_authors_short(["Radford", "Wu", "Child", "Luan"]) == "Radford et al."

    # get_paper_external_url
    assert get_paper_external_url("doi:10.1000/1") == "https://doi.org/10.1000/1"
    assert get_paper_external_url("arxiv:2401.00001") == "https://arxiv.org/abs/2401.00001"
    assert get_paper_external_url("W123456") is None
    p_with_url = Paper(paper_id="p1", title="Title", url="https://custom.org/paper.pdf")
    assert get_paper_external_url("p1", p_with_url) == "https://custom.org/paper.pdf"

    # format_finding_citations
    formatted = format_finding_citations("Claim statement [doi:10.1000/1].")
    assert "verdict-inline-citation" in formatted
    assert "https://doi.org/10.1000/1" in formatted


def test_design_system_and_responsive_tokens_consistency():
    """Verify that the established light theme tokens, narrow screen rules, and mobile navigation are consistent."""
    css_content = THEME_CSS_PATH.read_text(encoding="utf-8")

    # 1. Exact design tokens
    assert "#1E1B2E" in css_content  # Primary text
    assert "#6941C6" in css_content  # Accent
    assert "#5932B3" in css_content  # Accent hover
    assert "#F5F4F9" in css_content  # Main workspace
    assert "#FFFFFF" in css_content  # Cards
    assert "#F9F8FD" in css_content  # Soft surfaces
    assert "#E2E0EC" in css_content  # Borders
    assert "#525066" in css_content  # Secondary text
    assert "#6E6C80" in css_content  # Muted text

    # 2. Responsive mobile rules
    assert "@media (max-width: 480px)" in css_content
    assert "@media (max-width: 390px)" in css_content
    assert "safe-area-inset-bottom" in css_content
    assert "overflow-wrap: break-word" in css_content

    # 3. Navigation items alignment across sidebar and mobile
    assert len(SIDEBAR_NAV_ITEMS) == 5
    assert len(MOBILE_NAV_ITEMS) == 5
    sidebar_keys = [item[0] for item in SIDEBAR_NAV_ITEMS]
    mobile_keys = [item[0] for item in MOBILE_NAV_ITEMS]
    assert sidebar_keys == mobile_keys
