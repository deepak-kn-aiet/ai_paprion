"""Tests for the End-to-End Research Agent orchestrator (agent.agent).

All tests are isolated, deterministic, and mocked — no live network requests or external API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from agent.agent import (
    ResearchAgent,
    ResearchAgentError,
    ResearchAgentResult,
)
from agent.overview import OverviewEngine
from agent.planner import (
    LLMProviderError,
    LLMResponseError,
    QueryItem,
    QueryPlan,
    QueryPlanner,
)
from shared.models import Paper

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_paper(
    paper_id: str,
    title: str,
    abstract: str | None = "Sample empirical abstract.",
    **kwargs,
) -> Paper:
    """Create a valid Paper for agent orchestration testing."""
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=kwargs.get("year", 2024),
        venue=kwargs.get("venue", "NeurIPS"),
        authors=kwargs.get("authors", ["Alice", "Bob"]),
        citation_count=kwargs.get("citation_count", 50),
        doi=kwargs.get("doi", paper_id if paper_id.startswith("10.") else None),
        arxiv_id=kwargs.get("arxiv_id"),
        score=kwargs.get("score"),
        is_recommended=kwargs.get("is_recommended"),
    )


def _make_mock_search(paper_map: dict[str, list[Paper]] | None = None):
    """Return a mock search function that routes queries to predefined papers."""
    mapping = paper_map or {}

    def _search(query: str, **kwargs) -> list[Paper]:
        return mapping.get(query, [])

    return _search


# ---------------------------------------------------------------------------
# Topic Validation Tests
# ---------------------------------------------------------------------------


class TestAgentTopicValidation:
    """Tests input validation for the research topic."""

    def test_empty_string_topic_raises_value_error(self) -> None:
        agent = ResearchAgent(provider="mock", search_fn=_make_mock_search())
        with pytest.raises(ValueError, match="non-empty string"):
            agent.run(topic="")

    def test_whitespace_only_topic_raises_value_error(self) -> None:
        agent = ResearchAgent(provider="mock", search_fn=_make_mock_search())
        with pytest.raises(ValueError, match="non-empty string"):
            agent.run(topic="    \n\t  ")

    def test_non_string_topic_raises_value_error(self) -> None:
        agent = ResearchAgent(provider="mock", search_fn=_make_mock_search())
        with pytest.raises(ValueError, match="non-empty string"):
            agent.run(topic=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Successful End-to-End Orchestration Tests
# ---------------------------------------------------------------------------


class TestAgentEndToEndSuccess:
    """Tests valid end-to-end orchestration workflows."""

    def test_successful_orchestration_mock_mode(self) -> None:
        paper1 = _make_paper("doi:10.1000/1", "Foundations of Attention")
        paper2 = _make_paper("doi:10.1000/2", "Scaling Transformers")

        mock_search = MagicMock(return_value=[paper1, paper2])
        agent = ResearchAgent(provider="mock", search_fn=mock_search)

        result = agent.run(topic="Transformer Architectures")

        assert isinstance(result, ResearchAgentResult)
        assert result.topic == "Transformer Architectures"
        assert 3 <= len(result.plan.queries) <= 5
        assert result.total_retrieved > 0
        assert result.total_deduplicated > 0
        assert len(result.ranked_papers) > 0
        assert result.overview.verdict.confidence in ("high", "medium", "low")
        assert len(result.diagnostics) == len(result.plan.queries)
        assert all(d.status in ("success", "empty") for d in result.diagnostics)

    def test_deduplication_across_planned_queries(self) -> None:
        paper_a = _make_paper("doi:10.1000/1", "Unique Paper A")
        paper_b = _make_paper("doi:10.1000/2", "Shared Paper B")
        paper_c = _make_paper("doi:10.1000/3", "Unique Paper C")

        # Mock query plan with 3 fixed queries
        plan = QueryPlan(
            topic="Attention",
            queries=[
                QueryItem(query="attention foundations", intent="foundational"),
                QueryItem(query="attention benchmarks", intent="benchmarks"),
                QueryItem(query="attention applications", intent="applications"),
            ],
        )
        mock_planner = MagicMock()
        mock_planner.plan.return_value = plan

        # Query 1 returns A and B; Query 2 returns B (duplicate) and C; Query 3 returns B (duplicate)
        call_count = 0

        def _search(query: str, **kwargs) -> list[Paper]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [paper_a, paper_b]
            if call_count == 2:
                return [paper_b, paper_c]
            return [paper_b]

        agent = ResearchAgent(
            planner=mock_planner,
            overview_engine=OverviewEngine(provider="mock"),
            search_fn=_search,
        )

        result = agent.run(topic="Attention")

        # Total raw retrieved: 2 + 2 + 1 = 5
        assert result.total_retrieved == 5
        # Total deduplicated: A, B, C = 3
        assert result.total_deduplicated == 3
        assert len(result.ranked_papers) == 3
        assert {p.paper_id for p in result.ranked_papers} == {
            "doi:10.1000/1",
            "doi:10.1000/2",
            "doi:10.1000/3",
        }

    def test_ranking_order_and_score_propagation(self) -> None:
        # Paper 1 has high citation count and Tier 1 venue
        paper_high = _make_paper(
            "doi:10.1000/high",
            "High Impact Paper",
            venue="NeurIPS",
            citation_count=5000,
            year=2023,
        )
        # Paper 2 has zero citations and unknown venue
        paper_low = _make_paper(
            "doi:10.1000/low",
            "Low Impact Paper",
            venue="Unknown Workshop",
            citation_count=0,
            year=2015,
        )

        mock_search = MagicMock(return_value=[paper_low, paper_high])
        agent = ResearchAgent(provider="mock", search_fn=mock_search)

        result = agent.run(topic="Machine Learning")

        # Must be sorted in descending order of score
        assert len(result.ranked_papers) == 2
        assert result.ranked_papers[0].score is not None
        assert result.ranked_papers[1].score is not None
        assert result.ranked_papers[0].score >= result.ranked_papers[1].score
        assert result.ranked_papers[0].paper_id == "doi:10.1000/high"
        # Score breakdown must be populated
        assert result.ranked_papers[0].score_breakdown is not None
        assert "citations_per_year" in result.ranked_papers[0].score_breakdown

    def test_max_overview_papers_caps_candidate_evidence(self) -> None:
        papers = [_make_paper(f"doi:10.1000/{i}", f"Paper {i}") for i in range(10)]
        mock_search = MagicMock(return_value=papers)

        mock_overview_engine = MagicMock()
        mock_overview_engine.generate_overview.return_value = OverviewEngine(
            provider="mock"
        ).generate_overview(papers[:3], topic="Testing")

        agent = ResearchAgent(
            planner=QueryPlanner(provider="mock"),
            overview_engine=mock_overview_engine,
            search_fn=mock_search,
        )

        result = agent.run(topic="Testing", max_overview_papers=3)

        # Ranked papers contains all 10
        assert len(result.ranked_papers) == 10
        # Overview engine was called with only top 3
        passed_papers = mock_overview_engine.generate_overview.call_args[1]["papers"]
        assert len(passed_papers) == 3


# ---------------------------------------------------------------------------
# Short-Circuit & Missing Data Tests
# ---------------------------------------------------------------------------


class TestAgentShortCircuitsAndMissingData:
    """Tests for empty search results and papers lacking abstracts."""

    def test_empty_retrieval_triggers_deterministic_empty_overview(self) -> None:
        mock_search = MagicMock(return_value=[])
        agent = ResearchAgent(provider="mock", search_fn=mock_search)

        result = agent.run(topic="Unexplored Exotic Physics")

        assert result.total_retrieved == 0
        assert result.total_deduplicated == 0
        assert result.ranked_papers == []
        assert result.overview.total_papers_analyzed == 0
        assert result.overview.verdict.confidence == "low"
        assert "No research papers" in result.overview.verdict.verdict_statement
        # Diagnostics record 'empty' for each query
        assert all(d.status == "empty" for d in result.diagnostics)
        assert all(d.paper_count == 0 for d in result.diagnostics)

    def test_all_papers_missing_abstracts_triggers_metadata_verdict(self) -> None:
        paper1 = _make_paper("doi:10.1000/1", "Title Only One", abstract=None)
        paper2 = _make_paper("doi:10.1000/2", "Title Only Two", abstract="   ")

        mock_search = MagicMock(return_value=[paper1, paper2])
        agent = ResearchAgent(provider="mock", search_fn=mock_search)

        result = agent.run(topic="Metadata Only Field")

        assert result.total_deduplicated == 2
        assert result.overview.papers_with_abstracts == 0
        assert result.overview.verdict.confidence == "low"
        assert "strictly limited to paper titles" in result.overview.verdict.verdict_statement


# ---------------------------------------------------------------------------
# Error Handling & Partial Failure Tests
# ---------------------------------------------------------------------------


class TestAgentErrorsAndPartialFailures:
    """Tests error propagation and resilience to partial retrieval failures."""

    def test_planner_failure_raises_research_agent_error(self) -> None:
        mock_planner = MagicMock()
        mock_planner.plan.side_effect = LLMProviderError("LLM quota exceeded")

        agent = ResearchAgent(
            planner=mock_planner,
            overview_engine=OverviewEngine(provider="mock"),
            search_fn=_make_mock_search(),
        )

        with pytest.raises(ResearchAgentError, match="Query planning failed") as exc_info:
            agent.run(topic="AI Ethics")

        assert isinstance(exc_info.value.__cause__, LLMProviderError)

    def test_overview_failure_raises_research_agent_error(self) -> None:
        mock_overview = MagicMock()
        mock_overview.generate_overview.side_effect = LLMResponseError("Malformed JSON")

        paper = _make_paper("doi:10.1000/1", "Sample Paper")
        mock_search = MagicMock(return_value=[paper])

        agent = ResearchAgent(
            planner=QueryPlanner(provider="mock"),
            overview_engine=mock_overview,
            search_fn=mock_search,
        )

        with pytest.raises(ResearchAgentError, match="Overview synthesis failed") as exc_info:
            agent.run(topic="Deep Learning")

        assert isinstance(exc_info.value.__cause__, LLMResponseError)

    def test_partial_retrieval_failure_continues_with_successful_queries(self) -> None:
        paper = _make_paper("doi:10.1000/1", "Surviving Paper")

        plan = QueryPlan(
            topic="Robustness",
            queries=[
                QueryItem(query="query failing", intent="foundational"),
                QueryItem(query="query succeeding", intent="benchmarks"),
                QueryItem(query="query empty", intent="applications"),
            ],
        )
        mock_planner = MagicMock()
        mock_planner.plan.return_value = plan

        def _search_with_partial_failure(query: str, **kwargs) -> list[Paper]:
            if query == "query failing":
                raise ConnectionError("Provider network timeout")
            if query == "query succeeding":
                return [paper]
            return []

        agent = ResearchAgent(
            planner=mock_planner,
            overview_engine=OverviewEngine(provider="mock"),
            search_fn=_search_with_partial_failure,
        )

        result = agent.run(topic="Robustness")

        # The agent should not abort; it continues with successful query results
        assert result.total_deduplicated == 1
        assert len(result.ranked_papers) == 1
        assert result.ranked_papers[0].paper_id == "doi:10.1000/1"

        # Diagnostics reflect exact per-query statuses
        diag_map = {d.query: d for d in result.diagnostics}
        assert diag_map["query failing"].status == "failed"
        assert "Provider network timeout" in str(diag_map["query failing"].error_message)
        assert diag_map["query succeeding"].status == "success"
        assert diag_map["query succeeding"].paper_count == 1
        assert diag_map["query empty"].status == "empty"
        assert diag_map["query empty"].paper_count == 0


# ---------------------------------------------------------------------------
# Configuration & Dependency Injection Tests
# ---------------------------------------------------------------------------


class TestAgentConfigurationAndMockMode:
    """Tests dependency injection and zero-API-key mock mode."""

    def test_mock_mode_without_api_keys(self) -> None:
        with patch.object(config, "GEMINI_API_KEY", None):
            paper = _make_paper("doi:10.1000/1", "Offline Paper")
            mock_search = MagicMock(return_value=[paper])

            agent = ResearchAgent(provider="mock", search_fn=mock_search)
            result = agent.run(topic="Offline Analysis")

            assert result.topic == "Offline Analysis"
            assert len(result.ranked_papers) == 1

    def test_dependency_injection_no_live_network_calls(self) -> None:
        mock_planner = MagicMock()
        mock_planner.plan.return_value = QueryPlan(
            topic="DI Test",
            queries=[
                QueryItem(query="q1", intent="foundational"),
                QueryItem(query="q2", intent="benchmarks"),
                QueryItem(query="q3", intent="applications"),
            ],
        )

        mock_overview = MagicMock()
        mock_overview.generate_overview.return_value = OverviewEngine(
            provider="mock"
        ).generate_overview([], topic="DI Test")

        mock_search = MagicMock(return_value=[])

        agent = ResearchAgent(
            planner=mock_planner,
            overview_engine=mock_overview,
            search_fn=mock_search,
        )
        result = agent.run(topic="DI Test")

        mock_planner.plan.assert_called_once_with("DI Test")
        assert mock_search.call_count == 3
        mock_overview.generate_overview.assert_called_once()
        assert result.total_retrieved == 0
