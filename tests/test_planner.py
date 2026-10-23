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
    GroqClient,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    MockLLMClient,
    QueryItem,
    QueryPlan,
    QueryPlanner,
    _clean_and_parse_json,
    generate_deterministic_query_plan,
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


# ---------------------------------------------------------------------------
# GroqClient Tests (Mocked)
# ---------------------------------------------------------------------------


class TestGroqClient:
    """Tests for the GroqClient LLM client implementation."""

    def test_groq_missing_api_key_raises_config_error(self) -> None:
        with pytest.raises(LLMConfigurationError, match="Groq API key is required"):
            GroqClient(api_key=None)
        with pytest.raises(LLMConfigurationError, match="Groq API key is required"):
            GroqClient(api_key="")

    @patch("groq.Groq")
    def test_groq_client_init_and_generate_success(self, mock_groq_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "queries": [
                {"query": "groq survey foundations", "intent": "foundational"},
                {"query": "groq architecture algorithms", "intent": "methodology"},
                {"query": "groq benchmark evaluation", "intent": "benchmarks"},
            ]
        })
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = GroqClient(api_key="mock_groq_secret", model="llama-3.3-70b-versatile")
        output = client.generate_text("Research Prompt", system_instruction="System Instruction")

        assert "groq survey foundations" in output
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    @patch("groq.Groq")
    def test_groq_client_rate_limit_error_handling(self, mock_groq_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client

        from groq import RateLimitError
        rate_err = RateLimitError(
            message="Rate limit reached: 30 RPM exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit reached"}},
        )
        mock_client.chat.completions.create.side_effect = rate_err

        client = GroqClient(api_key="mock_groq_secret")
        with pytest.raises(LLMRateLimitError, match="quota or rate limit exceeded"):
            client.generate_text("Prompt")

    @patch("groq.Groq")
    def test_groq_client_empty_response_raises(self, mock_groq_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "   "
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        client = GroqClient(api_key="mock_groq_secret")
        with pytest.raises(LLMResponseError, match="empty response"):
            client.generate_text("Prompt")

    @patch("groq.Groq")
    def test_groq_never_exposes_api_key_in_error_message(self, mock_groq_class: MagicMock) -> None:
        secret_key = "super_secret_groq_api_token_xyz999"
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = ConnectionError("Connection refused by host")

        client = GroqClient(api_key=secret_key)
        with pytest.raises(LLMProviderError) as exc_info:
            client.generate_text("Prompt")

        assert secret_key not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Deterministic Planner Tests
# ---------------------------------------------------------------------------


class TestDeterministicPlanner:
    """Tests for deterministic query planning without LLM calls."""

    def test_deterministic_planner_generates_valid_query_plan(self) -> None:
        plan = generate_deterministic_query_plan("Retrieval-Augmented Generation")
        assert plan.topic == "Retrieval-Augmented Generation"
        assert 3 <= len(plan.queries) <= 5

        intents = [q.intent for q in plan.queries]
        assert "foundational" in intents
        assert "methodology" in intents
        assert "benchmarks" in intents
        assert "applications" in intents

        # Queries must be distinct
        unique_queries = {q.query.lower().strip() for q in plan.queries}
        assert len(unique_queries) == len(plan.queries)

    def test_deterministic_planner_empty_topic_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            generate_deterministic_query_plan("")
        with pytest.raises(ValueError, match="non-empty string"):
            generate_deterministic_query_plan("   ")

    def test_query_planner_deterministic_flag_makes_zero_llm_calls(self) -> None:
        mock_client = MagicMock()
        planner = QueryPlanner(client=mock_client, deterministic=True)
        plan = planner.plan("Mechanistic Interpretability")

        assert mock_client.generate_text.call_count == 0
        assert len(plan.queries) == 4
        assert "mechanistic interpretability" in plan.queries[0].query.lower()


# ---------------------------------------------------------------------------
# Provider Selection Tests
# ---------------------------------------------------------------------------


class TestProviderSelection:
    """Tests provider selection between Gemini, Groq, and Mock."""

    @patch("groq.Groq")
    def test_planner_selects_groq_provider(self, mock_groq_class: MagicMock) -> None:
        planner = QueryPlanner(provider="groq", api_key="test_key", deterministic=False)
        assert isinstance(planner.client, GroqClient)
        assert planner.client.model == config.GROQ_MODEL

    @patch("google.genai.Client")
    def test_planner_selects_gemini_provider(self, mock_genai_class: MagicMock) -> None:
        planner = QueryPlanner(provider="gemini", api_key="test_key", deterministic=False)
        assert isinstance(planner.client, GeminiClient)

    def test_planner_missing_groq_key_raises_config_error(self) -> None:
        with (
            patch.object(config, "GROQ_API_KEY", None),
            pytest.raises(LLMConfigurationError, match="Groq API key is required"),
        ):
            QueryPlanner(provider="groq", api_key=None)

    def test_planner_unsupported_provider_raises(self) -> None:
        with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
            QueryPlanner(provider="unsupported_provider")

