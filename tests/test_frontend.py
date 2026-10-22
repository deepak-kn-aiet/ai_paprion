"""Tests for the frontend modular UI components and application entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

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
    run_agent_search,
    to_progress_value,
)
from frontend.ui.diagnostics import render_planned_queries, render_score_analysis
from frontend.ui.footer import render_footer
from frontend.ui.header import render_hero, render_report_header
from frontend.ui.hero_shader import HERO_SHADER_HTML, render_hero_shader
from frontend.ui.metrics import render_research_metrics
from frontend.ui.paper_card import (
    render_metric_cards,
    render_paper_card,
    render_score_breakdown,
)
from frontend.ui.paper_library import render_paper_library
from frontend.ui.particles import SWARM_BACKGROUND_HTML, render_particle_background
from frontend.ui.sidebar import SIDEBAR_NAV_ITEMS, render_sidebar
from frontend.ui.theme import THEME_CSS_PATH, render_theme
from frontend.ui.tubes_cursor import TUBES_CURSOR_HTML, render_tubes_cursor
from frontend.ui.verdict import render_executive_overview, render_verdict_section
from tests.test_app import _make_sample_agent_result

FRONTEND_APP_PATH = Path(__file__).resolve().parent.parent / "frontend" / "app.py"


def test_frontend_exports():
    """Verify all expected symbols are exported by frontend."""
    assert callable(render_hero)
    assert callable(render_report_header)
    assert callable(render_research_metrics)
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
    assert len(SIDEBAR_NAV_ITEMS) == 5
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
    assert len(at.button) == 1
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
