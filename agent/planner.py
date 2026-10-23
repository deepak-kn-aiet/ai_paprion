"""LLM Query Planner — Member B (AI reasoning & query decomposition).

Generates 3 to 5 diverse, high-signal academic search queries from a research
topic or goal, spanning multiple research perspectives:
- Foundational / Survey
- Methodology / Architecture
- Benchmarks / Datasets
- Applications / Recent Advances

Provides a provider-agnostic interface with Gemini for live generation and
deterministic MockLLMClient for offline testing.
"""

from __future__ import annotations

import contextlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

import config

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QueryPlannerError(Exception):
    """Base exception for all query planner errors."""


class LLMConfigurationError(QueryPlannerError):
    """Raised when LLM configuration or credentials are missing or invalid."""


class LLMProviderError(QueryPlannerError):
    """Raised when the underlying LLM provider call fails."""


class LLMRateLimitError(LLMProviderError):
    """Raised when an LLM provider hits rate limits or quota exhaustion."""


class LLMResponseError(QueryPlannerError):
    """Raised when the LLM returns an unparseable, malformed, or invalid response."""


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class QueryItem(BaseModel):
    """A single planned academic search query with research intent and rationale."""

    query: str = Field(..., description="The academic search query string.")
    intent: str = Field(
        default="general",
        description="Research perspective: e.g. foundational, methodology, benchmarks, applications.",
    )
    rationale: str | None = Field(
        default=None,
        description="Brief explanation of why this query addresses the research goal.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if not cleaned:
            raise ValueError("Query string must be non-empty.")
        return cleaned

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        return cleaned if cleaned else "general"

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None


class QueryPlan(BaseModel):
    """Collection of 3 to 5 diverse queries generated for a research topic."""

    topic: str = Field(..., description="Original user research topic or question.")
    queries: list[QueryItem] = Field(
        ...,
        description="List of 3 to 5 distinct academic search queries.",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if not cleaned:
            raise ValueError("Topic must be a non-empty string.")
        return cleaned

    @model_validator(mode="after")
    def validate_queries(self) -> QueryPlan:
        query_list = self.queries
        if not (3 <= len(query_list) <= 5):
            raise ValueError(
                f"QueryPlan must contain between 3 and 5 queries, got {len(query_list)}."
            )

        seen_queries: set[str] = set()
        for item in query_list:
            norm = item.query.strip().lower()
            if norm in seen_queries:
                raise ValueError(f"Duplicate query detected: '{item.query}'")
            seen_queries.add(norm)

        return self

    @property
    def query_strings(self) -> list[str]:
        """Convenience property returning pure query strings."""
        return [q.query for q in self.queries]


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are an expert academic research strategist.
Your task is to decompose a given research topic into 3 to 5 distinct, high-signal academic search queries suitable for literature search engines (such as Semantic Scholar, OpenAlex, arXiv, and PubMed).

You MUST generate between 3 and 5 diverse queries covering distinct research perspectives:
1. Foundational / Survey: Foundational theory, early landmark papers, taxonomy, or comprehensive survey literature.
2. Methodology / Architecture: Specific core algorithms, models, loss functions, techniques, or architectural designs.
3. Benchmarks / Datasets: Empirical benchmarks, standard datasets, experimental evaluations, or comparative metrics.
4. Applications / Recent Advances: Real-world domain applications, modern extensions, state-of-the-art results, or future directions.

Requirements:
- Each query must be concise and effective for academic paper keyword/semantic search (2–8 terms).
- Do not repeat terms or create redundant queries; each query must be distinct.
- Return ONLY valid JSON adhering strictly to this schema:
{
  "queries": [
    {
      "query": "string (the academic search query)",
      "intent": "string (one of: foundational, methodology, benchmarks, applications, recent_advances)",
      "rationale": "string (brief justification of why this query addresses this perspective)"
    }
  ]
}
"""

PLANNER_USER_PROMPT_TEMPLATE = """Research Topic: {topic}

Generate 3 to 5 diverse academic search queries covering foundational concepts, methodology, benchmarks/datasets, and applications/recent advances. Return strictly JSON.
"""


# ---------------------------------------------------------------------------
# LLM Client Interface and Implementations
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract interface for LLM text generation."""

    @abstractmethod
    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        """Generate text from the LLM given a prompt and optional system instruction."""


class MockLLMClient(LLMClient):
    """Deterministic mock client for offline operation and automated testing."""

    def __init__(self, canned_response: str | None = None) -> None:
        self.canned_response = canned_response

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        if self.canned_response is not None:
            return self.canned_response

        # Extract topic from prompt if possible
        topic_match = re.search(r"Research Topic:\s*(.+)", prompt)
        topic = topic_match.group(1).strip() if topic_match else "research topic"

        # Generate 4 deterministic, diverse perspective queries
        mock_payload = {
            "queries": [
                {
                    "query": f"{topic} survey foundations overview",
                    "intent": "foundational",
                    "rationale": "Captures foundational literature and survey overviews.",
                },
                {
                    "query": f"{topic} neural architecture algorithms methods",
                    "intent": "methodology",
                    "rationale": "Captures technical mechanisms, models, and core methods.",
                },
                {
                    "query": f"{topic} benchmark dataset empirical evaluation",
                    "intent": "benchmarks",
                    "rationale": "Captures standard benchmarks, datasets, and comparative results.",
                },
                {
                    "query": f"{topic} practical applications recent advances",
                    "intent": "applications",
                    "rationale": "Captures domain applications and state-of-the-art advances.",
                },
            ]
        }
        return json.dumps(mock_payload)


class GeminiClient(LLMClient):
    """Client for Google Gemini models using the google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        if not api_key:
            raise LLMConfigurationError(
                "Gemini API key is required for live generation. "
                "Set GEMINI_API_KEY in environment/.env or use LLM_PROVIDER=mock for offline operation."
            )
        self.api_key = api_key
        self.model = model

        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        except Exception as err:
            raise LLMConfigurationError(f"Failed to initialize Gemini client: {err}") from err

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        try:
            from google.genai import types

            config_params: dict[str, Any] = {
                "temperature": 0.2,
                "response_mime_type": "application/json",
            }
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            config = types.GenerateContentConfig(**config_params)
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as err:
            err_msg = str(err).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "rate limit" in err_msg or "quota" in err_msg:
                raise LLMRateLimitError(
                    "Gemini API quota or rate limit exceeded. Please wait a moment before retrying, "
                    "or switch your LLM_PROVIDER in .env."
                ) from err
            raise LLMProviderError(f"Gemini API request failed: {err}") from err

        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise LLMResponseError("Gemini returned an empty response.")

        return str(text)


class GroqClient(LLMClient):
    """Client for Groq models using the official groq SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        if not api_key:
            raise LLMConfigurationError(
                "Groq API key is required when using LLM_PROVIDER=groq. "
                "Set GROQ_API_KEY in your environment or .env file."
            )
        self.api_key = api_key
        self.model = model

        try:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        except Exception as err:
            raise LLMConfigurationError(f"Failed to initialize Groq client: {err}") from err

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=4096,
            )
        except Exception as err:
            err_name = type(err).__name__
            status_code = getattr(err, "status_code", None)
            err_msg = str(err).lower()
            if err_name == "RateLimitError" or status_code == 429 or "rate limit" in err_msg or "quota" in err_msg:
                raise LLMRateLimitError(
                    "Groq API quota or rate limit exceeded. Please wait a moment before retrying, "
                    "or switch your LLM_PROVIDER (e.g. to 'gemini') in .env."
                ) from err
            raise LLMProviderError(f"Groq API request failed: {err}") from err

        choice = response.choices[0]
        text = choice.message.content
        if not text or not str(text).strip():
            raise LLMResponseError("Groq returned an empty response.")

        return str(text)


# ---------------------------------------------------------------------------
# Deterministic Query Planning
# ---------------------------------------------------------------------------


def generate_deterministic_query_plan(topic: str) -> QueryPlan:
    """Decompose a research topic deterministically into 4 diverse, high-signal academic queries.

    Produces distinct queries spanning foundational, methodology, benchmarks,
    and applications/recent advances without consuming LLM API quota.
    """
    clean_topic = topic.strip()
    if not clean_topic:
        raise ValueError("Topic must be a non-empty string.")

    query_items = [
        QueryItem(
            query=f"{clean_topic} survey foundations overview",
            intent="foundational",
            rationale=f"Captures foundational theory, seminal papers, and survey literature on {clean_topic}.",
        ),
        QueryItem(
            query=f"{clean_topic} neural architecture algorithms methods",
            intent="methodology",
            rationale=f"Captures core technical mechanisms, algorithmic formulations, and architectures for {clean_topic}.",
        ),
        QueryItem(
            query=f"{clean_topic} benchmark dataset empirical evaluation",
            intent="benchmarks",
            rationale=f"Captures standard empirical benchmarks, evaluation datasets, and comparative metrics for {clean_topic}.",
        ),
        QueryItem(
            query=f"{clean_topic} practical applications recent advances",
            intent="applications",
            rationale=f"Captures domain applications, real-world case studies, and state-of-the-art advances in {clean_topic}.",
        ),
    ]

    return QueryPlan(topic=clean_topic, queries=query_items)


# ---------------------------------------------------------------------------
# Query Planner
# ---------------------------------------------------------------------------


def _clean_and_parse_json(raw_text: str) -> dict[str, Any] | list[Any]:
    """Parse raw LLM output into JSON, stripping Markdown code fences if present."""
    cleaned = raw_text.strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    with contextlib.suppress(json.JSONDecodeError, ValueError):
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed

    # Fallback: search for first bracket/brace enclosure
    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    start_bracket = cleaned.find("[")
    end_bracket = cleaned.rfind("]")

    candidate = None
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        candidate = cleaned[start_brace : end_brace + 1]
    elif start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        candidate = cleaned[start_bracket : end_bracket + 1]

    if candidate:
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed

    raise LLMResponseError(
        f"Failed to parse LLM output as JSON. Output was:\n{raw_text[:300]}"
    )


class QueryPlanner:
    """High-level query planning engine that decomposes research topics into diverse queries."""

    def __init__(
        self,
        client: LLMClient | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        deterministic: bool | None = None,
    ) -> None:
        if client is not None:
            self.client = client
            self.deterministic = bool(deterministic) if deterministic is not None else False
            return

        resolved_provider = (provider or config.LLM_PROVIDER).strip().lower()
        self.deterministic = (
            deterministic
            if deterministic is not None
            else (False if resolved_provider in ("mock", "offline") else config.USE_DETERMINISTIC_PLANNER)
        )

        if resolved_provider in ("mock", "offline"):
            self.client = MockLLMClient()
        elif resolved_provider == "gemini":
            resolved_key = api_key or config.GEMINI_API_KEY
            resolved_model = model or config.LLM_MODEL
            self.client = GeminiClient(api_key=resolved_key, model=resolved_model)
        elif resolved_provider == "groq":
            resolved_key = api_key or config.GROQ_API_KEY
            resolved_model = model or config.GROQ_MODEL
            self.client = GroqClient(api_key=resolved_key, model=resolved_model)
        else:
            raise LLMConfigurationError(
                f"Unsupported LLM provider: '{resolved_provider}'. "
                "Supported providers are 'gemini', 'groq', and 'mock'."
            )

    def plan(self, topic: str) -> QueryPlan:
        """Decompose a research topic into 3 to 5 diverse academic search queries.

        Parameters
        ----------
        topic : str
            The user-supplied research topic, query, or goal. Must be a non-empty string.

        Returns
        -------
        QueryPlan
            Validated query plan with 3 to 5 distinct queries spanning multiple perspectives.

        Raises
        ------
        ValueError
            If topic is empty or not a string.
        LLMProviderError
            If live provider network call fails.
        LLMResponseError
            If response cannot be parsed or fails query constraints.
        """
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topic must be a non-empty string.")

        clean_topic = topic.strip()
        if self.deterministic:
            return generate_deterministic_query_plan(clean_topic)

        user_prompt = PLANNER_USER_PROMPT_TEMPLATE.format(topic=clean_topic)

        raw_response = self.client.generate_text(
            prompt=user_prompt,
            system_instruction=PLANNER_SYSTEM_PROMPT,
        )

        parsed_json = _clean_and_parse_json(raw_response)

        raw_queries: list[Any]
        if isinstance(parsed_json, dict):
            raw_queries = parsed_json.get("queries", [])
        elif isinstance(parsed_json, list):
            raw_queries = parsed_json
        else:
            raise LLMResponseError("JSON response structure must be an object with 'queries' or a list.")

        if not isinstance(raw_queries, list):
            raise LLMResponseError("Field 'queries' must be a list.")

        query_items: list[QueryItem] = []
        for idx, item in enumerate(raw_queries):
            try:
                if isinstance(item, str):
                    query_items.append(QueryItem(query=item, intent="general"))
                elif isinstance(item, dict):
                    query_items.append(QueryItem(**item))
                else:
                    raise LLMResponseError(
                        f"Query item #{idx} is invalid type: {type(item).__name__}"
                    )
            except Exception as err:
                raise LLMResponseError(
                    f"Invalid query item #{idx} in LLM response: {err}"
                ) from err

        try:
            return QueryPlan(topic=clean_topic, queries=query_items)
        except Exception as err:
            raise LLMResponseError(
                f"Generated queries failed QueryPlan validation: {err}"
            ) from err
