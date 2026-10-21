from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import (
    build_paper_card,
    build_paper_overview,
    build_paper_verdict,
    build_score_breakdown,
    compute_weighted_score,
    format_count,
    format_score,
    get_mock_results,
    is_valid_component_score,
    is_valid_topic,
    mock_search,
    to_progress_value,
)
from mock_data import SCORE_COMPONENT_WEIGHTS

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
STYLES_PATH = Path(__file__).resolve().parent.parent / "ui" / "styles.css"


def test_mock_results_have_required_fields():
    results = get_mock_results()
    assert len(results) == 3
    for result in results:
        assert {"title", "summary"} <= set(result.keys())
        assert result["title"]
        assert result["summary"]


@pytest.mark.parametrize("topic", ["", "   ", "\t", "\n", " \t \n "])
def test_blank_topic_is_rejected(topic):
    assert not is_valid_topic(topic)


def test_non_empty_topic_is_accepted():
    assert is_valid_topic("retrieval-augmented generation")
    assert is_valid_topic("  transformer models  ")


def test_mock_search_returns_demonstration_results():
    results = mock_search("transformer models")
    assert results == get_mock_results()
    assert len(results) > 0


def test_mock_search_returns_at_least_three_ranked_papers():
    results = mock_search("transformer models")
    assert len(results) >= 3
    ranks = [paper["rank"] for paper in results]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_mock_papers_have_ranking_fields_and_score_in_unit_range():
    for paper in get_mock_results():
        for field in (
            "rank",
            "title",
            "authors",
            "year",
            "venue",
            "citations",
            "influential_citations",
            "score",
        ):
            assert field in paper
        assert 0 <= paper["score"] <= 1


def test_build_paper_card_formats_all_required_fields():
    card = build_paper_card(get_mock_results()[0])
    assert card["rank"] == 1
    assert card["title"] == "Attention Is All You Need"
    assert "Ashish Vaswani" in card["authors"]
    assert card["year"] == 2017
    assert card["venue"] == "NeurIPS"
    assert card["citations"] == "148,000"
    assert card["influential_citations"] == "88,000"
    assert card["score"] == "0.97 / 1.0"
    assert card["paper_link"]
    assert card["pdf_link"]


def test_build_paper_card_handles_partial_links():
    card = build_paper_card(get_mock_results()[1])
    assert card["paper_link"]
    assert card["pdf_link"] is None


def test_build_paper_card_handles_missing_links_and_fields():
    paper = {"rank": 9, "title": "Minimal Paper"}
    card = build_paper_card(paper)
    assert card["authors"] == "Unknown authors"
    assert card["year"] == "N/A"
    assert card["venue"] == "N/A"
    assert card["citations"] == "N/A"
    assert card["influential_citations"] == "N/A"
    assert card["score"] == "N/A"
    assert card["paper_link"] is None
    assert card["pdf_link"] is None


def test_build_paper_card_handles_paper_without_any_links():
    card = build_paper_card(get_mock_results()[2])
    assert card["paper_link"] is None
    assert card["pdf_link"] is None


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


def test_build_paper_card_includes_score_value_for_progress_bar():
    card = build_paper_card(get_mock_results()[0])
    assert card["score_value"] == 0.97
    minimal = build_paper_card({"rank": 9, "title": "Minimal Paper"})
    assert minimal["score_value"] is None


def test_search_replaces_searching_message_with_completion_message():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("transformer models").run()
    at.button[0].click().run()
    assert len(at.info) == 0
    assert len(at.success) == 1
    assert at.success[0].value == "Mock search complete."


def test_search_still_renders_three_ranked_cards():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("transformer models").run()
    at.button[0].click().run()
    rank_headings = [
        element.value
        for element in at.markdown
        if str(element.value).startswith("**Rank #")
    ]
    assert len(rank_headings) == 3


def test_empty_topic_shows_warning_without_completion_message():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("   ").run()
    at.button[0].click().run()
    assert len(at.warning) == 1
    assert len(at.success) == 0


def test_warning_message_text_is_unchanged():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("   ").run()
    at.button[0].click().run()
    assert at.warning[0].value == "Please enter a research topic before searching."


@pytest.mark.parametrize(
    "kind,background,text",
    [
        ("Warning", "#fef3c7", "#78350f"),
        ("Success", "#ecfdf5", "#065f46"),
        ("Info", "#eef2ff", "#3730a3"),
        ("Error", "#fef2f2", "#991b1b"),
    ],
)
def test_alert_styles_use_readable_colors(kind, background, text):
    css = STYLES_PATH.read_text(encoding="utf-8")
    selector = f'[data-testid="stAlertContainer"]:has([data-testid="stAlertContent{kind}"])'
    assert selector in css
    block = css.split(selector, 1)[1].split("}", 1)[0]
    assert f"background: {background}" in block
    assert f"color: {text}" in block


def test_alert_styles_target_streamlit_alert_test_ids():
    css = STYLES_PATH.read_text(encoding="utf-8")
    assert 'data-testid="stAlertContainer"' in css
    assert "stAlertContentWarning" in css
    assert "stWarning" not in css
    assert "stSuccess" not in css


def test_score_component_weights_sum_to_one():
    assert sum(SCORE_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(SCORE_COMPONENT_WEIGHTS) == {
        "citation_impact",
        "influential_citation_ratio",
        "venue_quality",
        "recency",
        "review_score",
    }


def test_mock_component_scores_reproduce_existing_overall_scores():
    for paper in get_mock_results():
        breakdown = build_score_breakdown(paper)
        assert compute_weighted_score(breakdown) == pytest.approx(paper["score"])


def test_weighted_contributions_equal_component_score_times_weight():
    for paper in get_mock_results():
        for row in build_score_breakdown(paper):
            assert row["available"] is True
            assert 0.0 <= row["score"] <= 1.0
            assert row["contribution"] == pytest.approx(row["weight"] * row["score"])


def test_build_score_breakdown_marks_missing_components_unavailable():
    paper = {"score_components": {"citation_impact": 0.90, "review_score": 0.70}}
    breakdown = build_score_breakdown(paper)
    assert breakdown[0]["available"] is True
    assert breakdown[0]["contribution"] == pytest.approx(0.40 * 0.90)
    assert breakdown[1]["available"] is False
    assert breakdown[1]["score"] is None
    assert breakdown[1]["contribution"] is None


def test_compute_weighted_score_redistributes_weights_when_components_missing():
    breakdown = build_score_breakdown(
        {"score_components": {"citation_impact": 0.90, "review_score": 0.70}}
    )
    # Available weight is 0.40 + 0.10 = 0.50, redistributed over two components.
    assert compute_weighted_score(breakdown) == pytest.approx(0.86)


def test_compute_weighted_score_handles_missing_or_invalid_components():
    assert compute_weighted_score(build_score_breakdown({})) is None
    assert compute_weighted_score(build_score_breakdown({"score_components": {}})) is None
    invalid = build_score_breakdown({"score_components": {"recency": "high"}})
    assert compute_weighted_score(invalid) is None
    out_of_range = build_score_breakdown({"score_components": {"recency": 1.5}})
    assert compute_weighted_score(out_of_range) is None


def test_is_valid_component_score_enforces_unit_range():
    assert is_valid_component_score(0)
    assert is_valid_component_score(1)
    assert is_valid_component_score(0.5)
    assert not is_valid_component_score(-0.1)
    assert not is_valid_component_score(1.1)
    assert not is_valid_component_score(None)
    assert not is_valid_component_score("high")
    assert not is_valid_component_score(True)


def test_build_paper_card_includes_score_breakdown_and_weighted_total():
    card = build_paper_card(get_mock_results()[0])
    assert len(card["score_breakdown"]) == 5
    assert card["weighted_total"] == pytest.approx(0.97)
    minimal = build_paper_card({"rank": 9, "title": "Minimal Paper"})
    assert all(row["available"] is False for row in minimal["score_breakdown"])
    assert minimal["weighted_total"] is None


def test_search_renders_score_breakdown_for_each_paper():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("transformer models").run()
    at.button[0].click().run()
    breakdown_expanders = [
        expander
        for expander in at.expander
        if expander.label.startswith("Score breakdown (")
    ]
    assert len(breakdown_expanders) == 3
    breakdown_tables = [
        element
        for element in at.markdown
        if "Weighted contribution" in str(element.value)
    ]
    assert len(breakdown_tables) == 3
    for element in breakdown_tables:
        table = str(element.value)
        assert "Citation impact" in table
        assert "0.40" in table
        assert "Weighted total" in table


def test_mock_papers_have_overview_content():
    for paper in get_mock_results():
        overview = build_paper_overview(paper)
        assert overview["available"] is True
        assert overview["research_problem"]
        assert overview["main_contribution"]
        assert overview["related_topics"]


def test_build_paper_overview_handles_missing_fields():
    overview = build_paper_overview({})
    assert overview["available"] is False
    assert overview["research_problem"] is None
    assert overview["main_contribution"] is None
    assert overview["related_topics"] == []

    partial = build_paper_overview({"overview": {"research_problem": "Problem."}})
    assert partial["available"] is True
    assert partial["research_problem"] == "Problem."
    assert partial["main_contribution"] is None
    assert partial["related_topics"] == []


def test_build_paper_card_includes_topic_and_overview():
    card = build_paper_card(get_mock_results()[0], "transformer models")
    assert card["topic"] == "transformer models"
    assert card["overview"]["available"] is True
    default = build_paper_card({"rank": 1, "title": "Minimal Paper"})
    assert default["topic"] == ""
    assert default["overview"]["available"] is False


def test_search_renders_overview_expanders():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("retrieval-augmented generation").run()
    at.button[0].click().run()
    overview_labels = [
        expander.label
        for expander in at.expander
        if expander.label == "Paper overview (demonstration)"
    ]
    assert len(overview_labels) == 3
    problem_sections = [
        element
        for element in at.markdown
        if "Research problem" in str(element.value)
    ]
    assert len(problem_sections) == 3
    contribution_sections = [
        element
        for element in at.markdown
        if "Main contribution" in str(element.value)
    ]
    assert len(contribution_sections) == 3
    topic_notes = [
        caption
        for caption in at.caption
        if "not analyzed in this demonstration version" in str(caption.value)
    ]
    assert len(topic_notes) == 3
    assert any(
        "retrieval-augmented generation" in str(caption.value)
        for caption in topic_notes
    )


def test_mock_papers_have_verdict_content():
    for paper in get_mock_results():
        verdict = build_paper_verdict(paper)
        assert verdict["available"] is True
        assert verdict["recommendation"] in (
            "Keep this paper",
            "Consider another paper",
        )
        assert verdict["explanation"]


def test_build_paper_verdict_handles_missing_fields():
    verdict = build_paper_verdict({})
    assert verdict["available"] is False
    assert verdict["recommendation"] is None
    assert verdict["explanation"] is None

    partial = build_paper_verdict({"verdict": {"explanation": "Reason."}})
    assert partial["available"] is False
    assert partial["recommendation"] is None
    assert partial["explanation"] == "Reason."


def test_build_paper_card_includes_verdict():
    card = build_paper_card(get_mock_results()[0])
    assert card["verdict"]["available"] is True
    assert card["verdict"]["recommendation"] == "Keep this paper"
    minimal = build_paper_card({"rank": 1, "title": "Minimal Paper"})
    assert minimal["verdict"]["available"] is False


def test_search_renders_verdict_expanders():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("transformer models").run()
    at.button[0].click().run()
    verdict_expanders = [
        expander
        for expander in at.expander
        if expander.label == "Final verdict (demonstration)"
    ]
    assert len(verdict_expanders) == 3
    verdict_lines = [
        element
        for element in at.markdown
        if str(element.value).startswith("**Verdict:**")
    ]
    assert len(verdict_lines) == 3
    assert any(
        "Keep this paper" in str(element.value)
        for element in verdict_lines
    )
    assert any(
        "Consider another paper" in str(element.value)
        for element in verdict_lines
    )
    disclaimers = [
        caption
        for caption in at.caption
        if "not a genuine AI judgment" in str(caption.value)
    ]
    assert len(disclaimers) == 3
