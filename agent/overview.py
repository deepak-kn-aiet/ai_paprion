"""Paper Overview & Verdict Engine — Member B (AI reasoning & research synthesis).

Generates evidence-grounded research overviews, thematic clusters, and executive
verdicts from candidate research papers.

Guarantees & Methodological Boundaries:
- Evidence grounding: Claims and themes cite exact canonical paper_ids.
- Referential integrity vs. semantic proof: Automated citation checking validates
  referential integrity (confirming that cited paper IDs exist within the supplied
  candidate set and adhere to canonical formatting). It does not provide formal
  proof that a cited paper substantively justifies every nuanced assertion in the text.
- Distinction between abstract-supported and metadata-only papers: Papers with
  missing abstracts are explicitly flagged as metadata-only; substantive claims cannot
  be derived from titles alone.
- No hallucinations: Unsupported paper_ids, external prefixes, or index-style
  citations ([Paper 1], [1]) are strictly rejected with LLMResponseError.
- Deterministic handling for empty paper lists and missing abstracts.
- Cautious confidence calibration: Confidence evaluates the completeness,
  consistency, and coverage of the provided candidate evidence packet; it does
  not represent absolute scientific truth or certainty.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

import config
from agent.planner import (
    GeminiClient,
    GroqClient,
    LLMClient,
    LLMConfigurationError,
    LLMResponseError,
    _clean_and_parse_json,
)
from shared.models import Paper

# ---------------------------------------------------------------------------
# Pydantic Models for Overview and Verdict
# ---------------------------------------------------------------------------


class ResearchTheme(BaseModel):
    """A thematic cluster synthesized across candidate papers."""

    name: str = Field(..., description="Theme title, e.g. 'Sparse Attention Architectures'.")
    summary: str = Field(..., description="Concise synthesis of this theme across papers.")
    paper_ids: list[str] = Field(
        ...,
        description="Canonical paper_ids contributing to this theme. Must contain at least one ID.",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if not cleaned:
            raise ValueError("Theme name must be non-empty.")
        return cleaned

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if not cleaned:
            raise ValueError("Theme summary must be non-empty.")
        return cleaned

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, v: list[str]) -> list[str]:
        if not v or not isinstance(v, list):
            raise ValueError("Each ResearchTheme must cite at least one paper ID.")
        cleaned = [pid.strip() for pid in v if isinstance(pid, str) and pid.strip()]
        if not cleaned:
            raise ValueError("Each ResearchTheme must cite at least one non-empty paper ID.")
        return cleaned


class ResearchVerdict(BaseModel):
    """Executive verdict and consensus assessment."""

    verdict_statement: str = Field(
        ...,
        description="Executive takeaway summarizing consensus on the topic.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level grounded in evidence completeness and consistency.",
    )
    confidence_rationale: str = Field(
        ...,
        description="Justification based on paper counts, abstract availability, and consensus.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Core factual takeaways supported by citations.",
    )
    limitations_and_gaps: list[str] = Field(
        default_factory=list,
        description="Unaddressed assumptions, dataset biases, or open challenges.",
    )
    top_recommended_papers: list[str] = Field(
        default_factory=list,
        description="Canonical paper_ids of highest quality/relevance papers.",
    )

    @field_validator("verdict_statement")
    @classmethod
    def validate_verdict_statement(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if not cleaned:
            raise ValueError("Verdict statement must be non-empty.")
        return cleaned


class ResearchOverview(BaseModel):
    """Complete evidence-grounded research overview and verdict."""

    topic: str = Field(..., description="The user research topic or question analyzed.")
    total_papers_analyzed: int = Field(..., ge=0, description="Total candidate papers evaluated.")
    papers_with_abstracts: int = Field(..., ge=0, description="Number of papers with available abstracts.")
    themes: list[ResearchTheme] = Field(
        default_factory=list,
        description="Thematic clusters synthesized from the papers.",
    )
    verdict: ResearchVerdict = Field(
        ...,
        description="Executive verdict, confidence score, findings, and limitations.",
    )
    synthesis: str = Field(
        ...,
        description="Multi-paragraph cohesive narrative overview citing papers by paper_id.",
    )
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Fine-grained phase timings in seconds for evidence packet, synthesis, and validation.",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if not cleaned:
            raise ValueError("Topic must be a non-empty string.")
        return cleaned

    @field_validator("synthesis")
    @classmethod
    def validate_synthesis(cls, v: str) -> str:
        cleaned = v.strip() if isinstance(v, str) else ""
        if not cleaned:
            raise ValueError("Synthesis must be a non-empty string.")
        return cleaned

    @model_validator(mode="after")
    def validate_abstract_counts(self) -> ResearchOverview:
        if self.papers_with_abstracts > self.total_papers_analyzed:
            raise ValueError("papers_with_abstracts cannot exceed total_papers_analyzed.")
        return self


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

OVERVIEW_SYSTEM_PROMPT = """You are an expert academic research meta-reviewer and literature synthesist.
Your task is to analyze a packet of candidate research papers and generate an evidence-grounded overview and verdict.

CRITICAL EVIDENCE RULES:
1. Grounding: All claims, themes, and findings must be directly grounded in the provided paper packet.
2. Citations: Every theme, key finding, and recommendation MUST cite candidate papers using their exact canonical paper_id (e.g. [doi:...], [arxiv:...]).
   - Do NOT use generic numbers, shorthand, or index citations like '[Paper 1]', '[P1]', or '[1]'.
   - Do NOT invent external citation identifiers or prefixes.
3. No Fabrication: Do NOT invent methods, benchmarks, datasets, numerical results, statistics, citations, consensus, or external references not present in the packet.
4. Abstract vs. Metadata Distinction:
   - For papers with an abstract, base your synthesis of methods and findings on the abstract text.
   - For papers marked '[Abstract not available; metadata only]', do NOT invent methods or results; disclose that evidence for those papers is limited strictly to publication metadata.
5. Limitations & Gaps: Identify unaddressed assumptions, dataset biases, or open challenges reported across the papers.
6. Confidence Assessment:
   - Confidence reflects the completeness, consistency, and coverage of the supplied candidate evidence packet, not absolute scientific truth.
   - If few papers have abstracts, confidence must be 'low' or 'medium'.

Return ONLY valid JSON adhering strictly to this schema:
{
  "themes": [
    {
      "name": "string (Theme title)",
      "summary": "string (Synthesis across papers)",
      "paper_ids": ["string (exact canonical paper_id)"]
    }
  ],
  "verdict": {
    "verdict_statement": "string (Executive takeaway)",
    "confidence": "high" | "medium" | "low",
    "confidence_rationale": "string (Justification)",
    "key_findings": ["string (Finding citing [paper_id])"],
    "limitations_and_gaps": ["string (Limitations and evidence constraints)"],
    "top_recommended_papers": ["string (canonical paper_id)"]
  },
  "synthesis": "string (Cohesive narrative overview citing papers by [paper_id])"
}
"""

OVERVIEW_USER_PROMPT_TEMPLATE = """Research Topic: {topic}

Candidate Papers ({total_count} papers, {abstract_count} with abstract):
{evidence_packet}

Analyze the papers and return strictly valid JSON matching the schema.
"""


# ---------------------------------------------------------------------------
# Mock Overview Client
# ---------------------------------------------------------------------------


class OverviewMockClient(LLMClient):
    """Deterministic mock client tailored for research overview generation."""

    def __init__(self, canned_response: str | None = None) -> None:
        self.canned_response = canned_response

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        if self.canned_response is not None:
            return self.canned_response

        # Extract topic from prompt
        topic_match = re.search(r"Research Topic:\s*(.+)", prompt)
        topic = topic_match.group(1).strip() if topic_match else "research topic"

        # Extract canonical paper_ids from prompt
        paper_ids = re.findall(r"paper_id:\s*(\S+)", prompt)
        if not paper_ids:
            paper_ids = []

        half = max(1, len(paper_ids) // 2) if paper_ids else 0
        first_group = paper_ids[:half]
        second_group = paper_ids[half:] or first_group

        top_recs = paper_ids[: min(3, len(paper_ids))]

        mock_payload = {
            "themes": [
                {
                    "name": f"{topic} Core Methodologies",
                    "summary": f"Key algorithmic approaches and models identified across the literature for {topic}.",
                    "paper_ids": first_group,
                },
                {
                    "name": f"{topic} Empirical Benchmarks and Evaluations",
                    "summary": f"Evaluation datasets and benchmark assessments for {topic}.",
                    "paper_ids": second_group,
                },
            ],
            "verdict": {
                "verdict_statement": f"Substantial progress has been demonstrated in {topic}, with consistent empirical benchmarks.",
                "confidence": "high" if len(paper_ids) >= 6 else "medium" if len(paper_ids) >= 3 else "low",
                "confidence_rationale": f"Based on evaluation of {len(paper_ids)} candidate papers with available metadata.",
                "key_findings": [
                    f"Demonstrated empirical performance improvements across {topic} benchmarks [{paper_ids[0]}]."
                    if paper_ids
                    else f"Significant findings in {topic}.",
                    f"Adoption of modern architectural designs to address key scalability bottlenecks [{paper_ids[-1]}]."
                    if paper_ids
                    else f"Architectural scalability in {topic}.",
                ],
                "limitations_and_gaps": [
                    "Evaluations are primarily constrained to standardized benchmark datasets.",
                    "Long-term generalization and edge-case behaviors remain open research questions.",
                ],
                "top_recommended_papers": top_recs,
            },
            "synthesis": (
                f"A comprehensive review of research on '{topic}' highlights significant methodological innovation. "
                f"Early foundational formulations established theoretical bounds, while recent contributions focus on scalability. "
                "Further investigation is required to validate empirical robustness in out-of-distribution deployments."
            ),
        }
        return json.dumps(mock_payload)


# ---------------------------------------------------------------------------
# Overview Engine
# ---------------------------------------------------------------------------


def _format_evidence_packet(
    papers: list[Paper],
    max_abstract_chars: int | None = None,
) -> str:
    """Format candidate papers into a structured, evidence-grounded packet for the LLM.

    Applies a sensible context limit to prevent token exhaustion while retaining sufficient evidence.
    """
    limit = (
        max_abstract_chars
        if max_abstract_chars is not None
        else config.SYNTHESIS_MAX_ABSTRACT_CHARS
    )
    packets: list[str] = []
    for idx, paper in enumerate(papers, 1):
        if paper.abstract is not None and paper.abstract.strip():
            raw_abstract = paper.abstract.strip()
            if limit > 0 and len(raw_abstract) > limit:
                abstract_text = raw_abstract[:limit].rstrip() + "... [truncated]"
            else:
                abstract_text = raw_abstract
        else:
            abstract_text = "[Abstract not available; metadata only]"

        authors_str = ", ".join(paper.authors) if paper.authors else "Unknown"
        venue_str = paper.venue if paper.venue else "Unknown"
        year_str = str(paper.year) if paper.year is not None else "Unknown"
        citations_str = (
            str(paper.citation_count) if paper.citation_count is not None else "Unknown"
        )
        score_str = f"{paper.score:.3f}" if paper.score is not None else "Unscored"
        rec_str = str(paper.is_recommended) if paper.is_recommended is not None else "Unknown"

        packet = (
            f"[Paper {idx}]\n"
            f"paper_id: {paper.paper_id}\n"
            f"title: {paper.title}\n"
            f"year: {year_str}\n"
            f"venue: {venue_str}\n"
            f"authors: {authors_str}\n"
            f"citations: {citations_str}\n"
            f"score: {score_str}\n"
            f"is_recommended: {rec_str}\n"
            f"abstract: {abstract_text}\n"
        )
        packets.append(packet)

    return "\n".join(packets)


def _extract_bracketed_citations(text: str) -> list[str]:
    """Extract all citation tokens from bracketed expressions in text.

    Supports:
    - Single citations: [doi:10.1000/1]
    - Multi-paper citations separated by comma or semicolon:
      [doi:10.1000/1, doi:10.1000/2] or [doi:10.1000/1; arxiv:2401.00001]
    - DOIs with legitimate punctuation (parentheses, plus signs, internal semicolons):
      e.g. [doi:10.1002/(sici)1097-0142...3.0.co;2-i]
    - Index-style citations: [Paper 1], [P1], [1], [1, 2]
    - Arbitrary / external prefixes: [pmid:12345], [fake:999]
    """
    results: list[str] = []
    raw_brackets = re.findall(r"\[([^\[\]]+)\]", text)
    for raw in raw_brackets:
        cleaned_raw = raw.strip()
        if not cleaned_raw:
            continue

        # Check for index-style citations: e.g. [1], [Paper 1], [P1], [1, 2], [Paper 1; Paper 2]
        if re.match(
            r"^(?:paper\s*\d+|p\d+|\d+)(?:\s*[,;]\s*(?:paper\s*\d+|p\d+|\d+))*$",
            cleaned_raw,
            re.IGNORECASE,
        ):
            for part in re.split(r"[,;]\s*", cleaned_raw):
                part_clean = part.strip()
                if part_clean:
                    results.append(part_clean)
            continue

        # Split multiple citations separated by comma or semicolon.
        # Uses lookahead to avoid splitting legitimate semicolons inside DOIs (e.g. ';2-i'):
        # 1) delimiter followed by an identifier prefix and colon (e.g. ', doi:', '; arxiv:')
        # 2) delimiter followed by whitespace and an index-style citation (e.g. ', 2', '; Paper 2')
        delimiter_pattern = r"[,;]\s*(?=[a-zA-Z0-9_\-]+:\S)|[,;]\s+(?=(?:paper\s*\d+|p\d+|\d+)\b)"
        parts = re.split(delimiter_pattern, cleaned_raw, flags=re.IGNORECASE)
        for part in parts:
            part_clean = part.strip()
            if not part_clean:
                continue
            # Collect if it is a colon-prefixed identifier (e.g. doi:..., pmid:..., fake:...)
            # or an index citation (e.g. Paper 1, P1, 1).
            # Avoids incorrectly treating ordinary prose like [see Fig. 2], [Section 3], [preliminary result] as citations.
            if ":" in part_clean or re.match(r"^(?:paper\s*\d+|p\d+|\d+)$", part_clean, re.IGNORECASE):
                results.append(part_clean)

    return results


def _build_normalized_id_map(valid_paper_ids: set[str]) -> dict[str, str]:
    """Build lookup map normalizing case-insensitive identifier namespaces (DOIs, S2, OpenAlex).

    Preserves canonical paper_ids for output while safely accepting case variations
    according to identifier semantics.
    """
    mapping: dict[str, str] = {}
    for pid in valid_paper_ids:
        mapping[pid] = pid
        prefix, sep, val = pid.partition(":")
        if not sep:
            continue
        p_lower = prefix.lower()
        if p_lower in ("doi", "s2", "openalex"):
            # Fully case-insensitive namespace (DOI ISO 26324, hex/numeric S2, OpenAlex)
            mapping[pid.lower()] = pid
            mapping[f"{p_lower}:{val}"] = pid
        elif p_lower == "arxiv":
            # arXiv: prefix is case-insensitive, but identifier value may be case-sensitive
            mapping[f"arxiv:{val}"] = pid
            mapping[f"arXiv:{val}"] = pid
            mapping[f"ARXIV:{val}"] = pid
        else:
            # Generic namespace prefix case-insensitivity
            mapping[f"{p_lower}:{val}"] = pid
    return mapping


def _lookup_canonical_id(token: str, normalized_map: dict[str, str]) -> str | None:
    """Resolve a citation token to its canonical paper_id, respecting namespace semantics."""
    # 1. Exact match
    if token in normalized_map:
        return normalized_map[token]

    prefix, sep, val = token.partition(":")
    if not sep:
        return None

    p_lower = prefix.lower()
    # 2. Case-insensitive lookup for DOI, S2, OpenAlex
    if p_lower in ("doi", "s2", "openalex") and token.lower() in normalized_map:
        return normalized_map[token.lower()]

    # 3. Normalized prefix lookup for arXiv and others
    normalized_prefix_token = f"{p_lower}:{val}"
    if normalized_prefix_token in normalized_map:
        return normalized_map[normalized_prefix_token]

    return None


def _validate_citation_tokens(
    tokens: list[str],
    normalized_map: dict[str, str],
    field_name: str,
) -> None:
    """Validate that extracted citation tokens are valid and not uncanonical index citations."""
    for token in tokens:
        # Check for prohibited index-style citations
        if re.match(r"^(?:paper\s*\d+|p\d+|\d+)$", token, re.IGNORECASE):
            raise LLMResponseError(
                f"{field_name} used invalid index-style citation '[{token}]'. "
                "Citations must use exact canonical paper_ids (e.g. [doi:...]), not numeric or '[Paper X]' references."
            )

        # Validate colon-prefixed or canonical identifier tokens
        canon = _lookup_canonical_id(token, normalized_map)
        if canon is None:
            raise LLMResponseError(
                f"{field_name} cited unknown or unsupported paper_id '{token}' not present in candidate papers."
            )


def _calibrate_confidence(
    llm_confidence: str,
    papers_with_abstracts: int,
    total_papers: int,
    rationale: str,
) -> tuple[Literal["high", "medium", "low"], str]:
    """Cautious heuristic calibrating confidence against evidence completeness.

    Confidence Calibration Rules & Methodological Boundaries:
    - Confidence reflects the completeness, recency, and consensus of the supplied
      candidate evidence packet; it does not represent absolute scientific truth.
    - High confidence cannot be inferred merely from paper counts, citation counts,
      venue quality, or apparent consensus without sufficient abstract coverage.
    - Low: total_papers < 3, OR papers_with_abstracts == 0, OR abstract ratio < 0.50.
    - Medium: 3 <= papers_with_abstracts <= 5, with abstract ratio >= 0.50.
    - High: papers_with_abstracts >= 6, with abstract ratio >= 0.70.
    """
    ratio = (papers_with_abstracts / total_papers) if total_papers > 0 else 0.0

    # Determine maximum allowable confidence based on evidence completeness
    if total_papers < 3 or papers_with_abstracts == 0 or ratio < 0.50:
        max_allowed: Literal["high", "medium", "low"] = "low"
    elif papers_with_abstracts < 6 or ratio < 0.70:
        max_allowed = "medium"
    else:
        max_allowed = "high"

    conf_rank = {"low": 1, "medium": 2, "high": 3}
    requested_level = llm_confidence.lower().strip()
    if requested_level not in conf_rank:
        requested_level = "low"

    final_level: Literal["high", "medium", "low"]
    final_rationale = rationale

    if conf_rank[requested_level] > conf_rank[max_allowed]:
        final_level = max_allowed
        final_rationale = (
            f"{rationale} (Calibrated to '{max_allowed}' based on evidence completeness: "
            f"{papers_with_abstracts}/{total_papers} papers have abstracts)."
        )
    else:
        final_level = requested_level  # type: ignore[assignment]

    return final_level, final_rationale


class OverviewEngine:
    """Engine synthesizing candidate research papers into an overview and verdict."""

    def __init__(
        self,
        client: LLMClient | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        if client is not None:
            self.client = client
            return

        resolved_provider = (provider or config.LLM_PROVIDER).strip().lower()

        if resolved_provider in ("mock", "offline"):
            self.client = OverviewMockClient()
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

    def generate_overview(self, papers: list[Paper], topic: str) -> ResearchOverview:
        """Generate an evidence-grounded overview and verdict from candidate papers.

        Parameters
        ----------
        papers : list[Paper]
            Candidate research papers to synthesize (may be scored or raw).
        topic : str
            The user research topic or question. Must be a non-empty string.

        Returns
        -------
        ResearchOverview
            Validated synthesis containing thematic clusters, executive verdict,
            calibrated confidence rating, key findings, and limitations.

        Raises
        ------
        ValueError
            If topic is empty or papers is not a list.
        LLMResponseError
            If output is unparseable or references unsupported paper_ids.
        LLMProviderError
            If live provider network call fails.
        """
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topic must be a non-empty string.")
        if not isinstance(papers, list):
            raise TypeError("Papers must be a list.")

        clean_topic = topic.strip()

        # Deduplicate candidate papers by paper_id to avoid redundant evidence context
        seen_pids: set[str] = set()
        deduped_papers: list[Paper] = []
        for p in papers:
            if p.paper_id not in seen_pids:
                seen_pids.add(p.paper_id)
                deduped_papers.append(p)
        papers = deduped_papers
        total_papers = len(papers)

        # -------------------------------------------------------------------
        # Short-circuit 1: Zero candidate papers (no LLM call)
        # -------------------------------------------------------------------
        if total_papers == 0:
            return ResearchOverview(
                topic=clean_topic,
                total_papers_analyzed=0,
                papers_with_abstracts=0,
                themes=[],
                verdict=ResearchVerdict(
                    verdict_statement="No research papers were provided for analysis.",
                    confidence="low",
                    confidence_rationale="Insufficient evidence: 0 papers were provided.",
                    key_findings=[],
                    limitations_and_gaps=["No candidate papers provided for analysis."],
                    top_recommended_papers=[],
                ),
                synthesis="No candidate research papers were available to evaluate this topic.",
            )

        papers_with_abstracts = sum(
            1 for p in papers if p.abstract and p.abstract.strip()
        )

        valid_paper_ids = {p.paper_id for p in papers}

        # -------------------------------------------------------------------
        # Short-circuit 2: All candidate papers lack abstracts
        # Discloses evidence limitation deterministically without fabrication
        # -------------------------------------------------------------------
        if papers_with_abstracts == 0:
            top_ids = [p.paper_id for p in papers[: min(3, total_papers)]]
            return ResearchOverview(
                topic=clean_topic,
                total_papers_analyzed=total_papers,
                papers_with_abstracts=0,
                themes=[],
                verdict=ResearchVerdict(
                    verdict_statement=(
                        "Evidence is strictly limited to paper titles and publication metadata; "
                        "no abstracts are available to substantiate findings or methodologies."
                    ),
                    confidence="low",
                    confidence_rationale=(
                        f"Low confidence: 0 of {total_papers} candidate papers have an abstract available. "
                        "Substantive research claims cannot be evaluated from titles alone."
                    ),
                    key_findings=[],
                    limitations_and_gaps=[
                        (
                            "All candidate papers lack abstract text. Findings, methodology, and experimental results "
                            "cannot be reliably derived from titles alone."
                        ),
                    ],
                    top_recommended_papers=top_ids,
                ),
                synthesis=(
                    f"Analysis of {total_papers} papers on topic '{clean_topic}' was conducted based strictly on publication metadata. "
                    "None of the candidate papers have accessible abstract text. Consequently, no substantive methodologies, benchmarks, "
                    "or empirical findings could be verified without inventing evidence."
                ),
            )

        timings: dict[str, float] = {}

        # -------------------------------------------------------------------
        # Live / Mock LLM Synthesis
        # -------------------------------------------------------------------
        t_packet_0 = time.perf_counter()
        evidence_packet = _format_evidence_packet(papers)
        user_prompt = OVERVIEW_USER_PROMPT_TEMPLATE.format(
            topic=clean_topic,
            total_count=total_papers,
            abstract_count=papers_with_abstracts,
            evidence_packet=evidence_packet,
        )
        timings["evidence_packet_construction"] = round(time.perf_counter() - t_packet_0, 4)

        t_llm_0 = time.perf_counter()
        raw_response = self.client.generate_text(
            prompt=user_prompt,
            system_instruction=OVERVIEW_SYSTEM_PROMPT,
        )
        timings["llm_synthesis"] = round(time.perf_counter() - t_llm_0, 4)

        t_val_0 = time.perf_counter()
        parsed_json = _clean_and_parse_json(raw_response)
        if not isinstance(parsed_json, dict):
            raise LLMResponseError("JSON response structure must be an object containing 'themes' and 'verdict'.")

        raw_themes = parsed_json.get("themes")
        raw_verdict = parsed_json.get("verdict")
        raw_synthesis = parsed_json.get("synthesis")

        if not isinstance(raw_themes, list):
            raise LLMResponseError("Field 'themes' must be a list.")
        if not isinstance(raw_verdict, dict):
            raise LLMResponseError("Field 'verdict' must be a dictionary.")
        if not isinstance(raw_synthesis, str) or not raw_synthesis.strip():
            raise LLMResponseError("Field 'synthesis' must be a non-empty string.")

        # -------------------------------------------------------------------
        # Citation & Paper ID Grounding Validation (Requirement 5)
        # -------------------------------------------------------------------
        normalized_map = _build_normalized_id_map(valid_paper_ids)

        theme_items: list[ResearchTheme] = []
        for idx, theme_data in enumerate(raw_themes):
            if not isinstance(theme_data, dict):
                raise LLMResponseError(f"Theme item #{idx} is not an object.")

            theme_name = theme_data.get("name", "")
            if isinstance(theme_name, str):
                _validate_citation_tokens(
                    _extract_bracketed_citations(theme_name), normalized_map, f"Theme #{idx} name"
                )
            theme_summary = theme_data.get("summary", "")
            if isinstance(theme_summary, str):
                _validate_citation_tokens(
                    _extract_bracketed_citations(theme_summary), normalized_map, f"Theme '{theme_name}' summary"
                )

            theme_pids = theme_data.get("paper_ids", [])
            if not isinstance(theme_pids, list):
                raise LLMResponseError(f"Theme '{theme_name}' paper_ids must be a list.")

            canonical_theme_pids: list[str] = []
            for pid in theme_pids:
                if not isinstance(pid, str):
                    raise LLMResponseError(f"Theme '{theme_name}' cited non-string paper_id.")
                clean_pid = pid.strip()
                canon = _lookup_canonical_id(clean_pid, normalized_map)
                if canon is None:
                    raise LLMResponseError(
                        f"Theme '{theme_name}' cited unknown paper_id '{pid}' "
                        f"not present in candidate papers."
                    )
                canonical_theme_pids.append(canon)

            theme_copy = dict(theme_data)
            theme_copy["paper_ids"] = canonical_theme_pids
            try:
                theme_items.append(ResearchTheme(**theme_copy))
            except Exception as err:
                raise LLMResponseError(f"Validation failed for Theme #{idx}: {err}") from err

        # Validate top_recommended_papers in verdict
        raw_top_recs = raw_verdict.get("top_recommended_papers", [])
        if not isinstance(raw_top_recs, list):
            raise LLMResponseError("Field 'top_recommended_papers' must be a list.")
        canonical_top_recs: list[str] = []
        for pid in raw_top_recs:
            if not isinstance(pid, str):
                raise LLMResponseError("top_recommended_papers contained non-string paper_id.")
            clean_pid = pid.strip()
            canon = _lookup_canonical_id(clean_pid, normalized_map)
            if canon is None:
                raise LLMResponseError(
                    f"Verdict top_recommended_papers cited unknown paper_id '{pid}' "
                    f"not present in candidate papers."
                )
            canonical_top_recs.append(canon)

        # Validate cited paper_ids in key_findings
        raw_findings = raw_verdict.get("key_findings", [])
        if not isinstance(raw_findings, list):
            raise LLMResponseError("Field 'key_findings' must be a list.")
        for finding in raw_findings:
            if not isinstance(finding, str):
                continue
            finding_tokens = _extract_bracketed_citations(finding)
            _validate_citation_tokens(finding_tokens, normalized_map, "Finding")

        # Validate cited paper_ids in limitations_and_gaps
        raw_limitations = raw_verdict.get("limitations_and_gaps", [])
        if not isinstance(raw_limitations, list):
            raise LLMResponseError("Field 'limitations_and_gaps' must be a list.")
        for lim in raw_limitations:
            if not isinstance(lim, str):
                continue
            lim_tokens = _extract_bracketed_citations(lim)
            _validate_citation_tokens(lim_tokens, normalized_map, "Limitations and gaps")

        # Validate cited paper_ids in verdict_statement
        raw_v_stmt = raw_verdict.get("verdict_statement", "")
        if isinstance(raw_v_stmt, str):
            v_tokens = _extract_bracketed_citations(raw_v_stmt)
            _validate_citation_tokens(v_tokens, normalized_map, "Verdict statement")

        # Validate cited paper_ids in confidence_rationale
        raw_rationale_str = raw_verdict.get("confidence_rationale", "")
        if isinstance(raw_rationale_str, str):
            r_tokens = _extract_bracketed_citations(raw_rationale_str)
            _validate_citation_tokens(r_tokens, normalized_map, "Verdict confidence rationale")

        # Validate cited paper_ids in synthesis
        synthesis_tokens = _extract_bracketed_citations(raw_synthesis)
        _validate_citation_tokens(synthesis_tokens, normalized_map, "Synthesis")

        # -------------------------------------------------------------------
        # Confidence Calibration (Requirement 8)
        # -------------------------------------------------------------------
        raw_conf = raw_verdict.get("confidence", "low")
        raw_rationale = raw_verdict.get("confidence_rationale", "")
        calibrated_conf, calibrated_rationale = _calibrate_confidence(
            llm_confidence=str(raw_conf),
            papers_with_abstracts=papers_with_abstracts,
            total_papers=total_papers,
            rationale=str(raw_rationale),
        )

        verdict_dict: dict[str, Any] = dict(raw_verdict)
        verdict_dict["confidence"] = calibrated_conf
        verdict_dict["confidence_rationale"] = calibrated_rationale
        verdict_dict["top_recommended_papers"] = canonical_top_recs

        timings["citation_validation"] = round(time.perf_counter() - t_val_0, 4)

        try:
            verdict_obj = ResearchVerdict(**verdict_dict)
            return ResearchOverview(
                topic=clean_topic,
                total_papers_analyzed=total_papers,
                papers_with_abstracts=papers_with_abstracts,
                themes=theme_items,
                verdict=verdict_obj,
                synthesis=raw_synthesis.strip(),
                timings=timings,
            )
        except Exception as err:
            raise LLMResponseError(f"Validation failed for ResearchOverview: {err}") from err
