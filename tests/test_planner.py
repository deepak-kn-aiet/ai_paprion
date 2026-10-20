"""Tests for the LLM Query Planner module (agent.planner).

All tests are isolated, deterministic, and mocked — no live network calls or external API keys are required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import config
from agent.planner import (
    GeminiClient,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    MockLLMClient,
    QueryItem,
    QueryPlan,
    QueryPlanner,
    _clean_and_parse_json,
)

# ---------------------------------------------------------------------------
# QueryItem & QueryPlan Pydantic Validation Tests
# ---------------------------------------------------------------------------


class TestQueryModelsValidation:
    """Tests for QueryItem and QueryPlan Pydantic validation."""

    def test_query_item_valid(self) -> None:
        item = QueryItem(
            query="  graph neural networks drug discovery  ",
            intent="  methodology  ",
            rationale="  focuses on molecular graph methods  ",
        )
        assert item.query == "graph neural networks drug discovery"
        assert item.intent == "methodology"
        assert item.rationale == "focuses on molecular graph methods"

    def test_query_item_empty_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryItem(query="   ", intent="methodology")

    def test_query_item_default_intent(self) -> None:
        item = QueryItem(query="transformer attention")
        assert item.intent == "general"
        assert item.rationale is None

    def test_query_item_empty_intent_defaults_to_general(self) -> None:
        item = QueryItem(query="transformer attention", intent="   ")
        assert item.intent == "general"

    def test_query_item_empty_rationale_becomes_none(self) -> None:
        item = QueryItem(query="transformer attention", rationale="   ")
        assert item.rationale is None

    def test_query_plan_valid_range_of_queries(self) -> None:
        # 3 queries (minimum)
        plan_3 = QueryPlan(
            topic="Quantum Computing",
            queries=[
                QueryItem(query="q1"),
                QueryItem(query="q2"),
                QueryItem(query="q3"),
            ],
        )
        assert len(plan_3.queries) == 3
        assert plan_3.query_strings == ["q1", "q2", "q3"]

        # 4 queries
        plan_4 = QueryPlan(
            topic="Quantum Computing",
            queries=[
                QueryItem(query="q1"),
                QueryItem(query="q2"),
                QueryItem(query="q3"),
                QueryItem(query="q4"),
            ],
        )
        assert len(plan_4.queries) == 4

        # 5 queries (maximum)
        plan_5 = QueryPlan(
            topic="Quantum Computing",
            queries=[
                QueryItem(query="q1"),
                QueryItem(query="q2"),
                QueryItem(query="q3"),
                QueryItem(query="q4"),
                QueryItem(query="q5"),
            ],
        )
        assert len(plan_5.queries) == 5

    def test_query_plan_too_few_queries_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            QueryPlan(
                topic="AI Alignment",
                queries=[
                    QueryItem(query="q1"),
                    QueryItem(query="q2"),
                ],
            )
        assert "between 3 and 5 queries" in str(excinfo.value)

    def test_query_plan_too_many_queries_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            QueryPlan(
                topic="AI Alignment",
                queries=[QueryItem(query=f"query {i}") for i in range(6)],
            )
        assert "between 3 and 5 queries" in str(excinfo.value)

    def test_query_plan_duplicate_queries_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            QueryPlan(
                topic="Robotics",
                queries=[
                    QueryItem(query="reinforcement learning robotics"),
                    QueryItem(query="robotic manipulation grasping"),
                    QueryItem(query="reinforcement learning robotics"),
                ],
            )
        assert "Duplicate query detected" in str(excinfo.value)

    def test_query_plan_case_insensitive_duplicate_queries_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            QueryPlan(
                topic="Robotics",
                queries=[
                    QueryItem(query="Reinforcement Learning Robotics"),
                    QueryItem(query="robotic manipulation grasping"),
                    QueryItem(query="reinforcement learning robotics  "),
                ],
            )
        assert "Duplicate query detected" in str(excinfo.value)

    def test_query_plan_empty_topic_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryPlan(
                topic="   ",
                queries=[
                    QueryItem(query="q1"),
                    QueryItem(query="q2"),
                    QueryItem(query="q3"),
                ],
            )


# ---------------------------------------------------------------------------
# QueryPlanner Topic & Offline/Mock Operation Tests
# ---------------------------------------------------------------------------


class TestQueryPlannerMockAndValidation:
    """Tests for QueryPlanner input validation and MockLLMClient execution."""

    def test_plan_empty_topic_raises_value_error(self) -> None:
        planner = QueryPlanner(provider="mock")
        with pytest.raises(ValueError, match="non-empty string"):
            planner.plan("")

    def test_plan_whitespace_topic_raises_value_error(self) -> None:
        planner = QueryPlanner(provider="mock")
        with pytest.raises(ValueError, match="non-empty string"):
            planner.plan("   \t\n  ")

    def test_plan_non_string_topic_raises_value_error(self) -> None:
        planner = QueryPlanner(provider="mock")
        with pytest.raises(ValueError, match="non-empty string"):
            planner.plan(None)  # type: ignore[arg-type]

    def test_default_mock_planner_generates_diverse_queries(self) -> None:
        planner = QueryPlanner(provider="mock")
        plan = planner.plan("Multimodal Large Language Models")

        assert isinstance(plan, QueryPlan)
        assert plan.topic == "Multimodal Large Language Models"
        assert 3 <= len(plan.queries) <= 5

        intents = [q.intent for q in plan.queries]
        # Should cover distinct perspectives
        assert "foundational" in intents
        assert "methodology" in intents
        assert "benchmarks" in intents
        assert "applications" in intents

        for q in plan.queries:
            assert "Multimodal Large Language Models" in q.query
            assert q.rationale is not None

    def test_mock_planner_offline_alias(self) -> None:
        planner = QueryPlanner(provider="offline")
        plan = planner.plan("Graph Neural Networks")
        assert len(plan.queries) == 4

    def test_mock_planner_via_config(self) -> None:
        with patch.object(config, "LLM_PROVIDER", "mock"):
            planner = QueryPlanner()
            assert isinstance(planner.client, MockLLMClient)
            plan = planner.plan("Retrieval-Augmented Generation")
            assert len(plan.queries) == 4

    def test_mock_planner_with_canned_json(self) -> None:
        canned_data = {
            "queries": [
                {
                    "query": "diffusion models survey",
                    "intent": "foundational",
                    "rationale": "Comprehensive survey on diffusion techniques.",
                },
                {
                    "query": "score matching denoising sampling",
                    "intent": "methodology",
                    "rationale": "Core mathematical formulation.",
                },
                {
                    "query": "text-to-image synthesis benchmark",
                    "intent": "benchmarks",
                    "rationale": "Standard benchmarks for image generation.",
                },
            ]
        }
        mock_client = MockLLMClient(canned_response=json.dumps(canned_data))
        planner = QueryPlanner(client=mock_client)

        plan = planner.plan("Diffusion Models")
        assert len(plan.queries) == 3
        assert plan.queries[0].query == "diffusion models survey"
        assert plan.queries[0].intent == "foundational"


# ---------------------------------------------------------------------------
# Response Parsing & Fence Stripping Tests
# ---------------------------------------------------------------------------


class TestResponseParsingAndFormatting:
    """Tests for raw JSON, Markdown fences, and list formats."""

    def test_clean_and_parse_json_raw_object(self) -> None:
        raw = '{"queries": [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]}'
        res = _clean_and_parse_json(raw)
        assert isinstance(res, dict)
        assert "queries" in res

    def test_clean_and_parse_json_markdown_fenced(self) -> None:
        raw = (
            "```json\n"
            '{"queries": [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]}\n'
            "```"
        )
        res = _clean_and_parse_json(raw)
        assert isinstance(res, dict)
        assert len(res["queries"]) == 3

    def test_clean_and_parse_json_generic_fences(self) -> None:
        raw = (
            "```\n"
            '{"queries": [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]}\n'
            "```"
        )
        res = _clean_and_parse_json(raw)
        assert isinstance(res, dict)

    def test_clean_and_parse_json_surrounded_by_text(self) -> None:
        raw = (
            "Here are the planned queries for your research topic:\n\n"
            '{"queries": [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]}\n\n'
            "I hope this helps your research!"
        )
        res = _clean_and_parse_json(raw)
        assert isinstance(res, dict)
        assert len(res["queries"]) == 3

    def test_plan_with_json_array_format(self) -> None:
        canned_array = json.dumps([
            {"query": "quantum error correction surface codes", "intent": "methodology"},
            {"query": "fault tolerant quantum computing benchmarks", "intent": "benchmarks"},
            {"query": "topological quantum computing review", "intent": "foundational"},
        ])
        planner = QueryPlanner(client=MockLLMClient(canned_response=canned_array))
        plan = planner.plan("Quantum Computing")
        assert len(plan.queries) == 3
        assert plan.queries[0].query == "quantum error correction surface codes"

    def test_plan_with_list_of_strings_format(self) -> None:
        canned_strings = json.dumps([
            "transformer attention mechanism",
            "self attention linear complexity",
            "vision transformer image recognition",
            "transformer language model evaluation",
        ])
        planner = QueryPlanner(client=MockLLMClient(canned_response=canned_strings))
        plan = planner.plan("Transformers")
        assert len(plan.queries) == 4
        assert all(q.intent == "general" for q in plan.queries)


# ---------------------------------------------------------------------------
# Malformed Responses and Error Handling Tests
# ---------------------------------------------------------------------------


class TestMalformedResponsesAndErrors:
    """Tests error handling for invalid JSON, schema mismatches, and provider failures."""

    def test_plan_malformed_json_raises_llm_response_error(self) -> None:
        planner = QueryPlanner(client=MockLLMClient(canned_response="This is not valid JSON at all."))
        with pytest.raises(LLMResponseError, match="Failed to parse LLM output"):
            planner.plan("Robotics")

    def test_plan_too_few_queries_raises_llm_response_error(self) -> None:
        canned = json.dumps({"queries": [{"query": "only one query"}]})
        planner = QueryPlanner(client=MockLLMClient(canned_response=canned))
        with pytest.raises(LLMResponseError, match="between 3 and 5 queries"):
            planner.plan("Robotics")

    def test_plan_too_many_queries_raises_llm_response_error(self) -> None:
        canned = json.dumps({"queries": [{"query": f"query {i}"} for i in range(7)]})
        planner = QueryPlanner(client=MockLLMClient(canned_response=canned))
        with pytest.raises(LLMResponseError, match="between 3 and 5 queries"):
            planner.plan("Robotics")

    def test_plan_duplicate_queries_raises_llm_response_error(self) -> None:
        canned = json.dumps({
            "queries": [
                {"query": "reinforcement learning"},
                {"query": "deep reinforcement learning"},
                {"query": "reinforcement learning"},
            ]
        })
        planner = QueryPlanner(client=MockLLMClient(canned_response=canned))
        with pytest.raises(LLMResponseError, match="Duplicate query detected"):
            planner.plan("RL")

    def test_plan_invalid_query_item_type_raises_llm_response_error(self) -> None:
        canned = json.dumps({"queries": [123, 456, 789]})
        planner = QueryPlanner(client=MockLLMClient(canned_response=canned))
        with pytest.raises(LLMResponseError, match="invalid type"):
            planner.plan("RL")

    def test_plan_queries_field_not_a_list_raises_llm_response_error(self) -> None:
        canned = json.dumps({"queries": "not a list"})
        planner = QueryPlanner(client=MockLLMClient(canned_response=canned))
        with pytest.raises(LLMResponseError, match="must be a list"):
            planner.plan("RL")


# ---------------------------------------------------------------------------
# Gemini Client & Provider Configuration Tests
# ---------------------------------------------------------------------------


class TestGeminiClientAndProviderConfig:
    """Tests for GeminiClient initialization, credentials, and error wrapping."""

    def test_gemini_missing_api_key_raises_configuration_error(self) -> None:
        with (
            patch.object(config, "GEMINI_API_KEY", None),
            pytest.raises(LLMConfigurationError, match="Gemini API key is required"),
        ):
            GeminiClient(api_key=None)

    def test_planner_unsupported_provider_raises_configuration_error(self) -> None:
        with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
            QueryPlanner(provider="anthropic_unsupported")

    def test_planner_gemini_mode_without_key_raises_configuration_error(self) -> None:
        with (
            patch.object(config, "GEMINI_API_KEY", None),
            pytest.raises(LLMConfigurationError, match="Gemini API key is required"),
        ):
            QueryPlanner(provider="gemini", api_key=None)

    @patch("google.genai.Client")
    def test_gemini_client_successful_generation(self, mock_genai_client_class: MagicMock) -> None:
        mock_client_instance = MagicMock()
        mock_genai_client_class.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "queries": [
                {"query": "federated learning privacy", "intent": "foundational"},
                {"query": "differential privacy aggregation algorithms", "intent": "methodology"},
                {"query": "federated learning medical benchmarks", "intent": "benchmarks"},
            ]
        })
        mock_client_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="mock_secret_key", model="gemini-2.5-flash")
        planner = QueryPlanner(client=client)

        plan = planner.plan("Federated Learning")
        assert len(plan.queries) == 3
        assert plan.queries[0].query == "federated learning privacy"

        # Verify google-genai client was invoked
        mock_client_instance.models.generate_content.assert_called_once()

    @patch("google.genai.Client")
    def test_gemini_client_empty_response_raises_response_error(
        self, mock_genai_client_class: MagicMock
    ) -> None:
        mock_client_instance = MagicMock()
        mock_genai_client_class.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = "   "
        mock_client_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="mock_secret_key")
        with pytest.raises(LLMResponseError, match="empty response"):
            client.generate_text("Prompt")

    @patch("google.genai.Client")
    def test_gemini_client_provider_error_wrapped(
        self, mock_genai_client_class: MagicMock
    ) -> None:
        mock_client_instance = MagicMock()
        mock_genai_client_class.return_value = mock_client_instance

        mock_client_instance.models.generate_content.side_effect = ConnectionError("Network unreachable")

        client = GeminiClient(api_key="mock_secret_key")
        with pytest.raises(LLMProviderError, match="Gemini API request failed"):
            client.generate_text("Prompt")
