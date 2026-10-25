
"""Streamlit frontend application entry point for AI Research Workspace."""

from __future__ import annotations

import functools
import html

import streamlit as st

SEARCH_SUGGESTIONS: list[dict[str, str]] = [
    {
        "label": "RAG in Biomedicine",
        "query": "Retrieval-augmented generation for biomedical question answering",
    },
    {
        "label": "Explainability in LLMs",
        "query": "Mechanistic interpretability in large language models",
    },
    {
        "label": "Molecular Diffusion",
        "query": "Diffusion models for molecular graph generation",
    },
    {
        "label": "Renewable Energy Forecasting",
        "query": "Renewable energy forecasting and grid demand optimization",
    },
]

import agent.agent
from agent.agent import (
    ResearchAgent,
    ResearchAgentError,
    ResearchAgentResult,
)
from agent.planner import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
)
from data.search import search
from frontend.ui.dashboard import (
    EXAMPLE_TOPICS,
    render_empty_results_state,
    render_initial_dashboard,
)
from frontend.ui.diagnostics import (
    render_planned_queries,
    render_score_analysis,
)
from frontend.ui.footer import render_footer
from frontend.ui.formatters import (
    COMPONENT_KEY_ALIASES,
    SCORE_BREAKDOWN_NOTE,
    build_paper_card,
    build_paper_overview,
    build_paper_verdict,
    build_score_breakdown,
    compute_weighted_score,
    format_authors,
    format_count,
    format_score,
    is_valid_component_score,
    to_progress_value,
)
from frontend.ui.header import (
    render_hero,
    render_report_header,
    render_team_banner,
)
from frontend.ui.hero_shader import render_hero_shader
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
from frontend.ui.particles import render_particle_background
from frontend.ui.sidebar import (
    MOBILE_NAV_ITEMS,
    SIDEBAR_NAV_ITEMS,
    record_recent_search,
    render_mobile_navigation,
    render_sidebar,
)
from frontend.ui.theme import (
    THEME_CSS_PATH,
    load_theme_css,
    render_theme,
)
from frontend.ui.tubes_cursor import render_tubes_cursor
from frontend.ui.verdict import (
    format_synthesis_citations,
    render_executive_overview,
    render_verdict_section,
)

__all__ = [
    "COMPONENT_KEY_ALIASES",
    "EXAMPLE_TOPICS",
    "MOBILE_NAV_ITEMS",
    "SCORE_BREAKDOWN_NOTE",
    "SEARCH_SUGGESTIONS",
    "SIDEBAR_NAV_ITEMS",
    "THEME_CSS_PATH",
    "ResearchAgent",
    "ResearchAgentError",
    "ResearchAgentResult",
    "build_paper_card",
    "build_paper_overview",
    "build_paper_verdict",
    "build_score_breakdown",
    "compute_weighted_score",
    "format_authors",
    "format_count",
    "format_score",
    "format_synthesis_citations",
    "is_valid_component_score",
    "is_valid_topic",
    "load_theme_css",
    "record_recent_search",
    "render",
    "render_empty_results_state",
    "render_executive_overview",
    "render_footer",
    "render_hero",
    "render_hero_shader",
    "render_initial_dashboard",
    "render_library_page",
    "render_metric_cards",
    "render_mobile_navigation",
    "render_overview_page",
    "render_overview_visualizations",
    "render_paper_card",
    "render_paper_library",
    "render_particle_background",
    "render_planned_queries",
    "render_report_header",
    "render_research_metrics",
    "render_research_results",
    "render_score_analysis",
    "render_score_analysis_page",
    "render_score_breakdown",
    "render_search_page",
    "render_sidebar",
    "render_team_banner",
    "render_theme",
    "render_tubes_cursor",
    "render_verdict_page",
    "render_verdict_section",
    "run_agent_search",
    "to_progress_value",
]


def is_valid_topic(topic: str | None) -> bool:
    """Validate that the research topic is not empty or whitespace."""
    return bool(topic and topic.strip())


def run_agent_search(topic: str) -> ResearchAgentResult:
    """Run research using the real OpenAlex retrieval and Gemini pipeline."""
    clean_topic = topic.strip()

    if not clean_topic:
        raise ValueError("Topic must be a non-empty string.")

    openalex_search = functools.partial(
        search,
        providers=["openalex"],
    )

    agent_cls = agent.agent.ResearchAgent
    agent_instance = agent_cls(search_fn=openalex_search)

    return agent_instance.run(clean_topic)


def render_search_section() -> tuple[str, bool]:
    """Render the research search form, suggestion chips, and submission controls."""
    st.markdown(
        '<div id="search-section"></div>',
        unsafe_allow_html=True,
    )

    search_col, button_col = st.columns(
        [4.5, 1],
        vertical_alignment="bottom",
    )

    with search_col:
        topic_input = st.text_input(
            "Enter research question or academic topic",
            placeholder=(
                "e.g., Retrieval-augmented generation for "
                "biomedical question answering"
            ),
            key="research_topic_input",
            label_visibility="visible",
        )

    with button_col:
        run_clicked = st.button(
            "Run Research",
            use_container_width=True,
            type="primary",
        )

    chips_markup: list[str] = []
    for item in SEARCH_SUGGESTIONS:
        escaped_query = html.escape(item["query"]).replace("'", "\\'")
        escaped_label = html.escape(item["label"])
        chip_html = f"""<div class="search-chip" role="button" tabindex="0" title="{escaped_query}"
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
             }})()"><span class="chip-spark">✦</span><span>{escaped_label}</span></div>"""
        chips_markup.append(chip_html)

    st.markdown(
        f"""
        <div class="search-chips-container">
            <span class="search-chips-label">SUGGESTED INQUIRIES</span>
            <div class="search-chips-list">
                {"".join(chips_markup)}
            </div>
        </div>
        <script>
        (function() {{
            try {{
                var doc = (window.parent && window.parent.document) ? window.parent.document : document;
                var inp = doc.querySelector('[data-testid="stTextInput"] input');
                if (inp && !inp.__enter_attached) {{
                    inp.__enter_attached = true;
                    inp.addEventListener('keydown', function(e) {{
                        if (e.key === 'Enter') {{
                            var btn = doc.querySelector('.stButton > button') || doc.querySelector('[data-testid="stBaseButton-primary"]');
                            if (btn) btn.click();
                        }}
                    }});
                }}
            }} catch(e) {{}}
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    return topic_input, run_clicked


def handle_research_submission(topic_input: str) -> None:
    """Validate and execute a submitted research request."""
    if not is_valid_topic(topic_input):
        st.warning("Please enter a research topic before running research.")
        return

    clean_topic = topic_input.strip()
    record_recent_search(clean_topic)

    status_placeholder = st.empty()
    status_placeholder.markdown(
        f"""
        <div class="research-status-banner">
            <div class="status-banner-header">
                <span class="status-pulse-dot"></span>
                <strong>Autonomous Research Pipeline Active</strong>
            </div>
            <div class="status-banner-topic">Target Inquiry: <em>{html.escape(clean_topic)}</em></div>
            <div class="status-banner-steps">
                <span>① Query Planning</span> → <span>② OpenAlex Retrieval</span> → <span>③ Multi-Factor Scoring</span> → <span>④ Gemini Synthesis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner(
            "Planning queries, retrieving OpenAlex papers, "
            "scoring evidence, and synthesizing findings..."
        ):
            result = run_agent_search(clean_topic)

        if not result.ranked_papers:
            status_placeholder.empty()
            st.warning(
                "No papers were retrieved for this research topic. "
                "Try refining your research question."
            )
            return

        st.session_state["agent_result"] = result
        st.session_state["active_topic"] = clean_topic
        st.session_state["pending_navigation"] = "Research Overview"
        st.session_state["search_success_message"] = (
            f"Research pipeline complete: {result.total_retrieved} records retrieved, "
            f"{result.total_deduplicated} unique papers across "
            f"{len(result.plan.queries)} queries."
        )
        status_placeholder.empty()
        st.rerun()

    except LLMConfigurationError as cfg_error:
        status_placeholder.empty()
        err_msg = str(cfg_error)
        if "GROQ_API_KEY" in err_msg or "groq" in err_msg.lower():
            st.error(
                f"Configuration error: {err_msg}. "
                "Verify that GROQ_API_KEY is configured correctly in your local .env file."
            )
        else:
            st.error(
                "Configuration error: GEMINI_API_KEY is not configured or is invalid. "
                "Verify that GEMINI_API_KEY is configured correctly in your local .env file."
            )

    except LLMRateLimitError as rate_error:
        status_placeholder.empty()
        st.error(
            f"Rate limit exceeded: {rate_error}"
        )

    except (LLMProviderError, LLMResponseError) as llm_error:
        status_placeholder.empty()
        st.error(
            f"AI Provider error ({type(llm_error).__name__}). "
            "Check network connectivity and provider quota limits."
        )

    except ResearchAgentError as agent_error:
        status_placeholder.empty()
        root_cause = agent_error.__cause__ or agent_error

        if isinstance(root_cause, LLMConfigurationError):
            err_msg = str(root_cause)
            if "GROQ_API_KEY" in err_msg or "groq" in err_msg.lower():
                st.error(
                    f"Configuration error: {err_msg}. "
                    "Verify that GROQ_API_KEY is configured correctly in your local .env file."
                )
            else:
                st.error(
                    "Configuration error: GEMINI_API_KEY is not configured or is invalid. "
                    "Verify GEMINI_API_KEY in your local .env file."
                )
        elif isinstance(root_cause, LLMRateLimitError):
            st.error(
                f"Rate limit exceeded: {root_cause}"
            )
        elif isinstance(root_cause, (LLMProviderError, LLMResponseError)):
            st.error(
                f"AI Provider error ({type(root_cause).__name__}). "
                "Check network connectivity and provider quota limits."
            )
        else:
            st.error(
                "Research orchestration failed. Verify provider "
                "connectivity and try again."
            )

    except Exception as exc:  # noqa: BLE001
        status_placeholder.empty()
        st.error(
            f"Research failed ({type(exc).__name__}). "
            "Check the application logs for details."
        )


def render_search_page() -> None:
    """Render the Research Search view: hero, search inputs, suggestions, and initial workflow."""
    render_hero()
    topic_input, run_clicked = render_search_section()
    if run_clicked:
        handle_research_submission(topic_input)
    render_initial_dashboard()


def render_overview_page(result: ResearchAgentResult | None, topic: str) -> None:
    """Render the Research Overview view: report header, metrics, executive synthesis, and themes."""
    if result is None:
        render_empty_results_state("Research Overview")
        return

    success_msg = st.session_state.pop("search_success_message", None)
    if success_msg:
        st.success(success_msg)

    render_report_header(topic, result)
    render_research_metrics(result, result.overview)
    render_overview_visualizations(result)
    render_executive_overview(result.overview, result)


def render_library_page(result: ResearchAgentResult | None, topic: str) -> None:
    """Render the Paper Library view: ranked papers, filtering/sorting controls, paper cards, and links."""
    if result is None:
        render_empty_results_state("Paper Library")
        return

    render_paper_library(result.ranked_papers, topic)


def render_score_analysis_page(result: ResearchAgentResult | None) -> None:
    """Render the Score Analysis view: methodology, component weights, ranked comparisons, and queries."""
    if result is None:
        render_empty_results_state("Score Analysis")
        return

    render_score_analysis(result.ranked_papers)
    render_planned_queries(result)


def render_verdict_page(result: ResearchAgentResult | None) -> None:
    """Render the Research Verdict view: verdict statement, confidence badge, key findings, limitations, and recommendations."""
    if result is None:
        render_empty_results_state("Research Verdict")
        return

    render_verdict_section(result.overview, papers=result.ranked_papers)


def render_research_results() -> None:
    """Render full research results across all sections (maintained for backward compatibility)."""
    current_result: ResearchAgentResult | None = (
        st.session_state.get("agent_result")
    )
    active_topic: str = st.session_state.get("active_topic", "")

    if current_result is None:
        return

    if not current_result.ranked_papers:
        st.warning(
            "No papers were retrieved for this research topic. "
            "Try refining your research question."
        )
        return

    render_report_header(active_topic, current_result)
    render_research_metrics(current_result, current_result.overview)
    render_overview_visualizations(current_result)
    render_executive_overview(current_result.overview, current_result)
    render_verdict_section(current_result.overview, papers=current_result.ranked_papers)
    render_paper_library(current_result.ranked_papers, active_topic)
    render_score_analysis(current_result.ranked_papers)
    render_planned_queries(current_result)


def render() -> None:
    """Render the academic workspace and execute user research requests."""
    st.set_page_config(
        page_title="AI Research Workspace",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_theme()
    # Query parameter deep-linking support for browser automation and section review
    nav_param = st.query_params.get("nav")
    nav_labels = [item[0] for item in SIDEBAR_NAV_ITEMS]
    if nav_param and nav_param in nav_labels:
        st.session_state["sidebar_navigation"] = nav_param

    if st.query_params.get("demo") == "true" and st.session_state.get("agent_result") is None:
        from tests.test_app import _make_sample_agent_result
        st.session_state["agent_result"] = _make_sample_agent_result("Retrieval-Augmented Generation")
        st.session_state["active_topic"] = "Retrieval-Augmented Generation"
    elif "topic" in st.query_params and st.session_state.get("agent_result") is None:
        url_topic = st.query_params.get("topic", "").strip()
        if url_topic:
            handle_research_submission(url_topic)

    active_nav = render_sidebar()

    if "agent_result" not in st.session_state:
        st.session_state["agent_result"] = None

    if "active_topic" not in st.session_state:
        st.session_state["active_topic"] = ""

    agent_result: ResearchAgentResult | None = st.session_state.get("agent_result")
    active_topic: str = st.session_state.get("active_topic", "")

    if active_nav == "Research Search":
        render_search_page()
    elif active_nav == "Research Overview":
        render_overview_page(agent_result, active_topic)
    elif active_nav == "Paper Library":
        render_library_page(agent_result, active_topic)
    elif active_nav == "Score Analysis":
        render_score_analysis_page(agent_result)
    elif active_nav == "Research Verdict":
        render_verdict_page(agent_result)
    else:
        render_search_page()

    render_mobile_navigation()
    render_footer()


if __name__ == "__main__":
    render()
