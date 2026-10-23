"""Agent package — Member B: scoring, ranking, and LLM reasoning."""

from agent.agent import (
    ResearchAgent,
    ResearchAgentError,
    ResearchAgentResult,
    RetrievalDiagnostic,
)
from agent.overview import (
    OverviewEngine,
    OverviewMockClient,
    ResearchOverview,
    ResearchTheme,
    ResearchVerdict,
)
from agent.planner import (
    GeminiClient,
    GroqClient,
    LLMClient,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    MockLLMClient,
    QueryItem,
    QueryPlan,
    QueryPlanner,
    QueryPlannerError,
    generate_deterministic_query_plan,
)
from agent.scoring import (
    DEFAULT_VENUE_SCORES,
    DEFAULT_WEIGHTS,
    normalize_review_score,
    score_papers,
)

__all__ = [
    "DEFAULT_VENUE_SCORES",
    "DEFAULT_WEIGHTS",
    "GeminiClient",
    "GroqClient",
    "LLMClient",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponseError",
    "MockLLMClient",
    "OverviewEngine",
    "OverviewMockClient",
    "QueryItem",
    "QueryPlan",
    "QueryPlanner",
    "QueryPlannerError",
    "ResearchAgent",
    "ResearchAgentError",
    "ResearchAgentResult",
    "ResearchOverview",
    "ResearchTheme",
    "ResearchVerdict",
    "RetrievalDiagnostic",
    "generate_deterministic_query_plan",
    "normalize_review_score",
    "score_papers",
]
