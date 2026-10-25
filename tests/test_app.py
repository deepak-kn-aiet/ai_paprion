from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from agent.agent import (
    ResearchAgentResult,
    RetrievalDiagnostic,
)
from agent.overview import (
    ResearchOverview,
    ResearchTheme,
    ResearchVerdict,
)
from agent.planner import (
    LLMConfigurationError,
    LLMProviderError,
    QueryItem,
    QueryPlan,
)
from app import (
    build_paper_card,
    build_paper_overview,
    build_paper_verdict,
    build_score_breakdown,
    compute_weighted_score,
    format_count,
    format_score,
    is_valid_component_score,
    is_valid_topic,
    run_agent_search,
    to_progress_value,
)
from mock_data import SCORE_COMPONENT_WEIGHTS, get_mock_results
from shared.models import Paper

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _make_sample_agent_result(topic: str = "retrieval-augmented generation") -> ResearchAgentResult:
    """Construct a complete, deterministic ResearchAgentResult for offline UI testing."""
    paper1 = Paper(
        paper_id="doi:10.1234/test.rag1",
        title="Foundations of Retrieval-Augmented Generation",
        abstract="Comprehensive study of retriever-generator architectures and grounding.",
        year=2024,
        venue="NeurIPS",
        authors=["Alice Smith", "Bob Jones"],
        citation_count=150,
        influential_citation_count=45,
        score=0.88,
        score_breakdown={
            "citations_per_year": 0.85,
            "influential_citation_ratio": 0.40,
            "venue_quality": 0.95,
            "recency": 0.90,
            "peer_review": 0.80,
        },
        is_recommended=True,
        is_open_access=True,
        url="https://doi.org/10.1234/test.rag1",
        pdf_url="https://example.com/paper1.pdf",
        doi="10.1234/test.rag1",
    )
    paper2 = Paper(
        paper_id="doi:10.1234/test.rag2",
        title="Graph-Based RAG for Complex Reasoning",
        abstract="Exploring structural knowledge graph integration with LLMs.",
        year=2023,
        venue="ACL",
        authors=["Carol White"],
        citation_count=60,
        influential_citation_count=12,
        score=0.65,
        score_breakdown={
            "citations_per_year": 0.60,
            "influential_citation_ratio": 0.30,
            "venue_quality": 0.85,
            "recency": 0.70,
            "peer_review": 0.60,
        },
        is_recommended=False,
        is_open_access=False,
        url="https://doi.org/10.1234/test.rag2",
        pdf_url=None,
        doi="10.1234/test.rag2",
    )

    plan = QueryPlan(
        topic=topic,
        queries=[
            QueryItem(intent="foundational", query="retrieval augmented generation survey", rationale="broad survey"),
            QueryItem(intent="methodology", query="retriever generator architecture", rationale="system architecture"),
            QueryItem(intent="benchmarks", query="RAG QA evaluation benchmarks", rationale="evaluation metrics"),
        ],
    )

    diagnostics = [
        RetrievalDiagnostic(query="retrieval augmented generation survey", status="success", paper_count=10),
        RetrievalDiagnostic(query="retriever generator architecture", status="success", paper_count=10),
        RetrievalDiagnostic(query="RAG QA evaluation benchmarks", status="empty", paper_count=0),
    ]

    overview = ResearchOverview(
        topic=topic,
        total_papers_analyzed=2,
        papers_with_abstracts=2,
        synthesis="Retrieval-Augmented Generation bridges internal LLM parameters with dynamic knowledge stores [doi:10.1234/test.rag1].",
        themes=[
            ResearchTheme(
                name="Architectural Foundations",
                summary="Retriever-generator couplings enhance factual consistency [doi:10.1234/test.rag1].",
                paper_ids=["doi:10.1234/test.rag1"],
            ),
            ResearchTheme(
                name="Structural Enhancements",
                summary="Graph representations capture multi-hop relations [doi:10.1234/test.rag2].",
                paper_ids=["doi:10.1234/test.rag2"],
            ),
        ],
        verdict=ResearchVerdict(
            verdict_statement="RAG significantly mitigates hallucination with measurable accuracy gains across academic benchmarks.",
            confidence="high",
            confidence_rationale="Consistent findings across peer-reviewed publications with full abstract coverage.",
            key_findings=[
                "External context grounding reduces hallucination rates substantially [doi:10.1234/test.rag1].",
                "Graph structures assist in multi-step reasoning queries [doi:10.1234/test.rag2].",
            ],
            limitations_and_gaps=[
                "Retriever-generator misalignment remains an open challenge.",
                "Latency overhead during real-time retrieval.",
            ],
            top_recommended_papers=["doi:10.1234/test.rag1"],
        ),
    )

    return ResearchAgentResult(
        topic=topic,
        plan=plan,
        total_retrieved=20,
        total_deduplicated=2,
        ranked_papers=[paper1, paper2],
        overview=overview,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Topic Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", ["", "   ", "\t", "\n", " \t \n "])
def test_blank_topic_is_rejected(topic):
    assert not is_valid_topic(topic)


def test_non_empty_topic_is_accepted():
    assert is_valid_topic("retrieval-augmented generation")
    assert is_valid_topic("  transformer models  ")


# ---------------------------------------------------------------------------
# Formatting & Score Helper Tests
# ---------------------------------------------------------------------------


def test_format_count_and_format_score_handle_invalid_values():
    assert format_count(None) == "N/A"
    assert format_count("many") == "N/A"
    assert format_count(1234) == "1,234"
    assert format_score(None) == "N/A"
    assert format_score("high") == "N/A"
    assert format_score(0.5) == "0.50 / 1.0"


def test_to_progress_value_clamps_and_handles_invalid():
    assert to_progress_value(0.97) == 0.97
    assert to_progress_value(1.5) == 1.0
    assert to_progress_value(-0.2) == 0.0
    assert to_progress_value(1) == 1.0
    assert to_progress_value(0) == 0.0
    assert to_progress_value(None) is None
    assert to_progress_value("high") is None


def test_score_component_weights_sum_to_one():
    assert sum(SCORE_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(SCORE_COMPONENT_WEIGHTS) == {
        "citation_impact",
        "influential_citation_ratio",
        "venue_quality",
        "recency",
        "review_score",
    }


def test_is_valid_component_score_enforces_unit_range():
    assert is_valid_component_score(0)
    assert is_valid_component_score(1)
    assert is_valid_component_score(0.5)
    assert not is_valid_component_score(-0.1)
    assert not is_valid_component_score(1.1)
    assert not is_valid_component_score(None)
    assert not is_valid_component_score("high")
    assert not is_valid_component_score(True)


# ---------------------------------------------------------------------------
# Score Breakdown & Weighted Total Tests
# ---------------------------------------------------------------------------


def test_build_score_breakdown_with_paper_model():
    paper = Paper(
        paper_id="doi:10.1234/test",
        title="Test Paper",
        score=0.82,
        score_breakdown={
            "citations_per_year": 0.80,
            "influential_citation_ratio": 0.50,
            "venue_quality": 0.90,
            "recency": 1.0,
            "peer_review": 0.70,
        },
    )
    breakdown = build_score_breakdown(paper)
    assert len(breakdown) == 5
    component_map = {row["component"]: row for row in breakdown}
    assert component_map["Citation impact"]["score"] == 0.80
    assert component_map["Citation impact"]["weight"] == 0.40
    assert component_map["Citation impact"]["contribution"] == pytest.approx(0.32)


def test_compute_weighted_score_redistributes_weights_when_components_missing():
    paper = Paper(
        paper_id="doi:10.1234/test",
        title="Test",
        score_breakdown={"citations_per_year": 0.90, "peer_review": 0.70},
    )
    breakdown = build_score_breakdown(paper)
    # Available weight is 0.40 (citations) + 0.10 (peer_review) = 0.50
    # Expected: (0.40 * 0.90 + 0.10 * 0.70) / 0.50 = 0.43 / 0.50 = 0.86
    assert compute_weighted_score(breakdown) == pytest.approx(0.86)


def test_compute_weighted_score_handles_missing_or_invalid_components():
    assert compute_weighted_score(build_score_breakdown({})) is None
    assert compute_weighted_score(build_score_breakdown(Paper(paper_id="1", title="T"))) is None


# ---------------------------------------------------------------------------
# Paper Card Mapping Tests
# ---------------------------------------------------------------------------


def test_build_paper_card_with_real_paper_model():
    paper = Paper(
        paper_id="doi:10.1234/test.paper",
        title="Real Academic Paper on RAG",
        abstract="Detailed research abstract explaining findings.",
        year=2024,
        venue="NeurIPS",
        authors=["Alice Smith", "Bob Jones"],
        citation_count=120,
        influential_citation_count=45,
        score=0.88,
        score_breakdown={
            "citations_per_year": 0.85,
            "influential_citation_ratio": 0.375,
            "venue_quality": 1.0,
            "recency": 0.9,
            "peer_review": 0.8,
        },
        is_recommended=True,
        is_open_access=True,
        url="https://doi.org/10.1234/test.paper",
        pdf_url="https://example.com/paper.pdf",
        doi="10.1234/test.paper",
    )

    card = build_paper_card(paper, topic="medical QA", rank=1)
    assert card["rank"] == 1
    assert card["title"] == "Real Academic Paper on RAG"
    assert card["authors"] == "Alice Smith, Bob Jones"
    assert card["year"] == 2024
    assert card["venue"] == "NeurIPS"
    assert card["citations"] == "120"
    assert card["influential_citations"] == "45"
    assert card["score"] == "0.88 / 1.0"
    assert card["score_value"] == 0.88
    assert card["summary"] == "Detailed research abstract explaining findings."
    assert card["paper_link"] == "https://doi.org/10.1234/test.paper"
    assert card["pdf_link"] == "https://example.com/paper.pdf"
    assert card["is_recommended"] is True
    assert card["is_open_access"] is True
    assert card["doi"] == "10.1234/test.paper"
    assert card["verdict"]["available"] is True
    assert card["verdict"]["recommendation"] == "Keep this paper"


def test_build_paper_card_with_real_paper_missing_optional_fields():
    minimal_paper = Paper(
        paper_id="arxiv:2401.00001",
        title="Minimal Paper",
    )

    card = build_paper_card(minimal_paper, rank=2)
    assert card["rank"] == 2
    assert card["title"] == "Minimal Paper"
    assert card["authors"] == "Unknown authors"
    assert card["year"] == "N/A"
    assert card["venue"] == "N/A"
    assert card["citations"] == "N/A"
    assert card["influential_citations"] == "N/A"
    assert card["score"] == "N/A"
    assert card["score_value"] is None
    assert card["summary"] == ""
    assert card["paper_link"] is None
    assert card["pdf_link"] is None
    assert card["verdict"]["available"] is False


def test_build_paper_card_with_fixture_dictionary():
    fixture = get_mock_results()[0]
    card = build_paper_card(fixture, rank=1)
    assert card["title"] == fixture["title"]
    assert card["year"] == fixture["year"]
    assert card["venue"] == fixture["venue"]
    assert len(card["score_breakdown"]) == 5


# ---------------------------------------------------------------------------
# Paper Overview and Verdict Helper Tests
# ---------------------------------------------------------------------------


def test_build_paper_overview_handles_missing_fields():
    overview = build_paper_overview({})
    assert overview["available"] is False
    assert overview["research_problem"] is None


def test_build_paper_verdict_handles_missing_fields():
    verdict = build_paper_verdict({})
    assert verdict["available"] is False
    assert verdict["recommendation"] is None


# ---------------------------------------------------------------------------
# run_agent_search Orchestrator Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_topic", ["", "   ", "\t", "\n", " \t \n "])
def test_run_agent_search_rejects_empty_or_whitespace_topic(empty_topic):
    with patch("app.ResearchAgent") as mock_agent_cls:
        with pytest.raises(ValueError, match="non-empty string"):
            run_agent_search(empty_topic)
        mock_agent_cls.assert_not_called()


def test_run_agent_search_invokes_agent_with_openalex():
    with patch("app.ResearchAgent") as mock_agent_cls:
        mock_instance = mock_agent_cls.return_value
        mock_instance.run.return_value = "agent_run_result"

        result = run_agent_search("retrieval augmented generation")
        assert result == "agent_run_result"
        mock_agent_cls.assert_called_once()
        _, kwargs = mock_agent_cls.call_args
        assert "search_fn" in kwargs
        search_fn = kwargs["search_fn"]
        assert search_fn.keywords == {"providers": ["openalex"]}
        mock_instance.run.assert_called_once_with("retrieval augmented generation")


# ---------------------------------------------------------------------------
# Streamlit AppTest End-to-End UI Tests (Offline with Mocked Agent)
# ---------------------------------------------------------------------------


def test_app_initial_render_without_search():
    """Verify application renders clean workspace header, input, and empty library when no search has run."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.title) >= 1
    assert at.title[0].value == "AI Research Workspace"
    assert len(at.text_input) == 1
    assert at.text_input[0].placeholder.startswith("e.g., Retrieval-augmented")
    assert len(at.button) >= 1
    assert at.button[0].label == "Run Research"
    # No error or warning on initial clean load
    assert len(at.error) == 0
    assert len(at.warning) == 0


def test_search_with_blank_topic_shows_warning():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input[0].input("   ")
    at.button[0].click()
    at.run()
    assert len(at.warning) == 1
    assert "Please enter a research topic" in str(at.warning[0].value)


def test_search_renders_real_agent_result_sections():
    """Verify that when ResearchAgent succeeds, pages switch correctly and render dedicated views."""
    sample_result = _make_sample_agent_result("Applications of RAG")
    with patch("agent.agent.ResearchAgent") as mock_agent_cls:
        mock_instance = mock_agent_cls.return_value
        mock_instance.run.return_value = sample_result

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input[0].input("Applications of RAG")
        at.button[0].click()
        at.run(timeout=30)

        # 1. Landed on Research Overview
        assert at.sidebar.radio[0].value == "Research Overview"
        assert len(at.success) >= 1
        assert "Research pipeline complete" in str(at.success[0].value)
        subheaders = [s.value for s in at.subheader]
        assert "Executive Research Overview" in subheaders
        assert any("Research Themes" in el for el in [e.label for e in at.expander])

        # 2. Switch to Paper Library
        at.sidebar.radio[0].set_value("Paper Library").run(timeout=30)
        lib_subheaders = [s.value for s in at.subheader]
        assert any("Paper Library" in s for s in lib_subheaders)
        rank_markdowns = [m.value for m in at.markdown if "Rank #" in str(m.value)]
        assert len(rank_markdowns) == 2

        # 3. Switch to Score Analysis
        at.sidebar.radio[0].set_value("Score Analysis").run(timeout=30)
        score_subheaders = [s.value for s in at.subheader]
        assert "Score Analysis & Methodology" in score_subheaders
        assert any("Planned Search Queries" in el for el in [e.label for e in at.expander])

        # 4. Switch to Research Verdict
        at.sidebar.radio[0].set_value("Research Verdict").run(timeout=30)
        verdict_subheaders = [s.value for s in at.subheader]
        assert "Research Verdict & Findings" in verdict_subheaders

        # 5. Switch back to Research Search
        at.sidebar.radio[0].set_value("Research Search").run(timeout=30)
        assert len(at.text_input) == 1
        assert len(at.button) >= 1


def test_empty_results_state_and_return_button():
    """Verify that visiting results pages before searching displays empty state and return button."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    for page in ["Research Overview", "Paper Library", "Score Analysis", "Research Verdict"]:
        at.sidebar.radio[0].set_value(page).run()
        markdown_text = " ".join(str(m.value) for m in at.markdown)
        assert f"No Active Research for {page}" in markdown_text
        return_btns = [b for b in at.button if "← Go to Research Search" in b.label]
        assert len(return_btns) == 1

    # Clicking return button takes user back to Research Search
    return_btns[0].click().run()
    assert at.sidebar.radio[0].value == "Research Search"
    assert len(at.text_input) == 1


def test_page_switching_preserves_results_without_agent_reinvocation():
    """Verify switching between pages preserves agent result and does not rerun agent."""
    sample_result = _make_sample_agent_result("Test Preservation")
    with patch("agent.agent.ResearchAgent") as mock_agent_cls:
        mock_instance = mock_agent_cls.return_value
        mock_instance.run.return_value = sample_result

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.text_input[0].input("Test Preservation")
        at.button[0].click().run()

        assert mock_instance.run.call_count == 1
        assert at.sidebar.radio[0].value == "Research Overview"

        # Cycle through all pages
        pages = ["Paper Library", "Score Analysis", "Research Verdict", "Research Search", "Research Overview"]
        for page in pages:
            at.sidebar.radio[0].set_value(page).run()
            # Agent must never be called again
            assert mock_instance.run.call_count == 1
            # Session state result must remain preserved
            assert at.session_state["agent_result"] is not None



def test_search_handles_empty_retrieved_papers():
    """Verify that when 0 papers are returned, a clear warning is shown."""
    empty_result = _make_sample_agent_result()
    empty_result.ranked_papers = []
    with patch("agent.agent.ResearchAgent") as mock_agent_cls:
        mock_instance = mock_agent_cls.return_value
        mock_instance.run.return_value = empty_result

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input[0].input("obscure topic without papers")
        at.button[0].click()
        at.run(timeout=30)

        assert len(at.warning) >= 1
        assert "No papers were retrieved" in str(at.warning[0].value)


def test_search_handles_llm_configuration_error():
    """Verify LLMConfigurationError displays clear guidance without exposing keys."""
    with patch("agent.agent.ResearchAgent") as mock_agent_cls:
        mock_instance = mock_agent_cls.return_value
        mock_instance.run.side_effect = LLMConfigurationError("API key missing")

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input[0].input("RAG in medicine")
        at.button[0].click()
        at.run(timeout=30)

        assert len(at.error) == 1
        assert "GEMINI_API_KEY is not configured" in str(at.error[0].value)


def test_search_handles_llm_provider_error():
    """Verify LLMProviderError surfaces a user-friendly error banner."""
    with patch("agent.agent.ResearchAgent") as mock_agent_cls:
        mock_instance = mock_agent_cls.return_value
        mock_instance.run.side_effect = LLMProviderError("Quota exceeded")

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input[0].input("RAG in medicine")
        at.button[0].click()
        at.run(timeout=30)

        assert len(at.error) == 1
        assert "AI Provider error" in str(at.error[0].value)


def test_mobile_navigation_tabs_exist_and_switch():
    """Verify that all 5 mobile navigation tabs are rendered and switch sections."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    mob_btns = [b for b in at.button if b.key and b.key.startswith("mob_nav_tab_")]
    assert len(mob_btns) == 5
    assert "Search" in mob_btns[0].label
    assert "Overview" in mob_btns[1].label
    assert "Library" in mob_btns[2].label
    assert "Analysis" in mob_btns[3].label
    assert "Verdict" in mob_btns[4].label

    # Click overview tab
    mob_btns[1].click().run()
    assert at.session_state["sidebar_navigation"] == "Research Overview"


def test_mobile_nav_items_export_contract():
    """Regression test: verify all three import paths and root application import."""
    import app
    assert hasattr(app, "MOBILE_NAV_ITEMS")

    from app import MOBILE_NAV_ITEMS as root_mobile_nav
    from frontend.app import MOBILE_NAV_ITEMS as frontend_mobile_nav
    from frontend.ui.sidebar import MOBILE_NAV_ITEMS as sidebar_mobile_nav
    from ui.sidebar import MOBILE_NAV_ITEMS as ui_mobile_nav

    assert root_mobile_nav == frontend_mobile_nav == sidebar_mobile_nav == ui_mobile_nav
    assert len(root_mobile_nav) == 5
    expected_keys = [
        "Research Search",
        "Research Overview",
        "Paper Library",
        "Score Analysis",
        "Research Verdict",
    ]
    assert [item[0] for item in root_mobile_nav] == expected_keys
    assert [item[0] for item in frontend_mobile_nav] == expected_keys
    assert [item[0] for item in sidebar_mobile_nav] == expected_keys
