"""Tests for the Paper Overview & Verdict module (agent.overview).

All tests are isolated, deterministic, and mocked — no live network calls or external API keys are required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import config
from agent.overview import (
    OverviewEngine,
    ResearchOverview,
    ResearchTheme,
    ResearchVerdict,
    _calibrate_confidence,
    _extract_bracketed_citations,
    _format_evidence_packet,
)
from agent.planner import (
    GeminiClient,
    LLMClient,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    MockLLMClient,
)
from shared.models import Paper

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_paper(paper_id: str, title: str, abstract: str | None = "Sample abstract text.", **kwargs) -> Paper:
    """Construct a minimal valid Paper for overview testing."""
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=kwargs.get("year", 2024),
        venue=kwargs.get("venue", "NeurIPS"),
        authors=kwargs.get("authors", ["Alice", "Bob"]),
        citation_count=kwargs.get("citation_count", 42),
        score=kwargs.get("score", 0.85),
        is_recommended=kwargs.get("is_recommended", True),
    )


# ---------------------------------------------------------------------------
# Pydantic Model Validation Tests
# ---------------------------------------------------------------------------


class TestOverviewModelsValidation:
    """Tests for ResearchTheme, ResearchVerdict, and ResearchOverview models."""

    def test_research_theme_valid(self) -> None:
        theme = ResearchTheme(
            name="  Sparse Attention Architectures  ",
            summary="  Techniques reducing quadratic transformer complexity.  ",
            paper_ids=["doi:10.1000/1", "arxiv:2401.00001"],
        )
        assert theme.name == "Sparse Attention Architectures"
        assert theme.summary == "Techniques reducing quadratic transformer complexity."
        assert len(theme.paper_ids) == 2

    def test_research_theme_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchTheme(name="   ", summary="Some summary.", paper_ids=["doi:10.1000/1"])

    def test_research_theme_empty_summary_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchTheme(name="Theme Name", summary="   ", paper_ids=["doi:10.1000/1"])

    def test_research_theme_empty_paper_ids_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one paper ID"):
            ResearchTheme(name="Theme Name", summary="Theme Summary", paper_ids=[])

    def test_research_theme_whitespace_paper_ids_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one non-empty paper ID"):
            ResearchTheme(name="Theme Name", summary="Theme Summary", paper_ids=["   ", ""])

    def test_research_verdict_valid(self) -> None:
        verdict = ResearchVerdict(
            verdict_statement="Transformers remain the dominant architecture.",
            confidence="high",
            confidence_rationale="Extensive empirical validation across benchmarks.",
            key_findings=["Linear attention scales to million-token contexts."],
            limitations_and_gaps=["High memory bandwidth requirements."],
            top_recommended_papers=["doi:10.1000/1"],
        )
        assert verdict.confidence == "high"
        assert len(verdict.key_findings) == 1

    def test_research_verdict_empty_statement_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchVerdict(
                verdict_statement="   ",
                confidence="low",
                confidence_rationale="Rationale",
            )

    def test_research_overview_abstracts_exceeding_total_raises(self) -> None:
        verdict = ResearchVerdict(
            verdict_statement="Valid verdict.",
            confidence="medium",
            confidence_rationale="Valid rationale.",
        )
        with pytest.raises(ValidationError, match="cannot exceed"):
            ResearchOverview(
                topic="Transformers",
                total_papers_analyzed=3,
                papers_with_abstracts=5,  # Exceeds total
                verdict=verdict,
                synthesis="Synthesis text.",
            )


# ---------------------------------------------------------------------------
# Evidence Packet Formatting & Confidence Calibration Tests
# ---------------------------------------------------------------------------


class TestEvidenceFormattingAndCalibration:
    """Tests for prompt formatting and cautious confidence heuristic."""

    def test_format_evidence_packet_distinguishes_abstracts(self) -> None:
        papers = [
            _make_paper("doi:10.1000/1", "Paper with Abstract", abstract="Detailed findings."),
            _make_paper("doi:10.1000/2", "Paper without Abstract", abstract=None),
        ]
        packet = _format_evidence_packet(papers)
        assert "Detailed findings." in packet
        assert "[Abstract not available; metadata only]" in packet
        assert "paper_id: doi:10.1000/1" in packet
        assert "paper_id: doi:10.1000/2" in packet

    def test_calibrate_confidence_sparse_abstracts_clamped_to_low(self) -> None:
        # LLM reports "high", but only 1 of 4 papers has an abstract
        conf, rationale = _calibrate_confidence(
            llm_confidence="high",
            papers_with_abstracts=1,
            total_papers=4,
            rationale="LLM confident.",
        )
        assert conf == "low"
        assert "Calibrated to 'low'" in rationale

    def test_calibrate_confidence_zero_abstracts_clamped_to_low(self) -> None:
        conf, _ = _calibrate_confidence(
            llm_confidence="medium",
            papers_with_abstracts=0,
            total_papers=5,
            rationale="LLM confident.",
        )
        assert conf == "low"

    def test_calibrate_confidence_moderate_evidence_clamped_to_medium(self) -> None:
        # 4 papers with abstracts: maximum allowed is medium
        conf, _ = _calibrate_confidence(
            llm_confidence="high",
            papers_with_abstracts=4,
            total_papers=5,
            rationale="LLM confident.",
        )
        assert conf == "medium"

    def test_calibrate_confidence_strong_evidence_allows_high(self) -> None:
        # 8 papers with abstracts: high is preserved
        conf, rationale = _calibrate_confidence(
            llm_confidence="high",
            papers_with_abstracts=8,
            total_papers=10,
            rationale="High consensus.",
        )
        assert conf == "high"
        assert "Calibrated" not in rationale


# ---------------------------------------------------------------------------
# Short-Circuit Tests (Zero Papers & Missing Abstracts)
# ---------------------------------------------------------------------------


class TestOverviewShortCircuits:
    """Tests for deterministic handling without unnecessary LLM invocation."""

    def test_empty_papers_list_returns_immediate_overview(self) -> None:
        mock_client = MagicMock(spec=LLMClient)
        engine = OverviewEngine(client=mock_client)

        overview = engine.generate_overview(papers=[], topic="Quantum Computing")
        assert overview.total_papers_analyzed == 0
        assert overview.papers_with_abstracts == 0
        assert overview.verdict.confidence == "low"
        assert overview.themes == []
        assert overview.verdict.key_findings == []
        assert "No research papers" in overview.verdict.verdict_statement
        # Zero LLM calls should have occurred
        mock_client.generate_text.assert_not_called()

    def test_all_missing_abstracts_returns_metadata_limitation_overview(self) -> None:
        mock_client = MagicMock(spec=LLMClient)
        engine = OverviewEngine(client=mock_client)

        papers = [
            _make_paper("doi:10.1000/1", "Title One", abstract=None),
            _make_paper("doi:10.1000/2", "Title Two", abstract="   "),
        ]
        overview = engine.generate_overview(papers=papers, topic="Deep Learning")
        assert overview.total_papers_analyzed == 2
        assert overview.papers_with_abstracts == 0
        assert overview.verdict.confidence == "low"
        assert overview.themes == []
        assert overview.verdict.key_findings == []
        assert "strictly limited to paper titles" in overview.verdict.verdict_statement
        assert "lack abstract text" in overview.verdict.limitations_and_gaps[0]
        # Zero LLM calls should have occurred
        mock_client.generate_text.assert_not_called()

    def test_empty_topic_raises_value_error(self) -> None:
        engine = OverviewEngine(provider="mock")
        with pytest.raises(ValueError, match="non-empty string"):
            engine.generate_overview(papers=[], topic="   ")

    def test_invalid_papers_type_raises_type_error(self) -> None:
        engine = OverviewEngine(provider="mock")
        with pytest.raises(TypeError, match="must be a list"):
            engine.generate_overview(papers=None, topic="Topic")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mock Overview & Valid Synthesis Tests
# ---------------------------------------------------------------------------


class TestOverviewGenerationMock:
    """Tests for valid overview generation with MockLLMClient and canned JSON."""

    def test_default_mock_engine_generates_valid_overview(self) -> None:
        engine = OverviewEngine(provider="mock")
        papers = [
            _make_paper("doi:10.1000/1", "Attention Is All You Need"),
            _make_paper("doi:10.1000/2", "BERT: Pre-training of Deep Bidirectional Transformers"),
            _make_paper("doi:10.1000/3", "RoBERTa: A Robustly Optimized BERT Pretraining Approach"),
            _make_paper("doi:10.1000/4", "Exploring the Limits of Transfer Learning with T5"),
        ]

        overview = engine.generate_overview(papers=papers, topic="Transformer Architectures")
        assert overview.topic == "Transformer Architectures"
        assert overview.total_papers_analyzed == 4
        assert overview.papers_with_abstracts == 4
        assert len(overview.themes) >= 1
        assert overview.verdict.confidence in ("high", "medium", "low")
        assert len(overview.verdict.key_findings) >= 1
        assert len(overview.verdict.top_recommended_papers) >= 1

    def test_canned_json_overview_succeeds(self) -> None:
        papers = [
            _make_paper("doi:10.1000/1", "Paper One"),
            _make_paper("doi:10.1000/2", "Paper Two"),
            _make_paper("doi:10.1000/3", "Paper Three"),
        ]
        canned_data = {
            "themes": [
                {
                    "name": "Foundational Advances",
                    "summary": "Core structural improvements.",
                    "paper_ids": ["doi:10.1000/1", "doi:10.1000/2"],
                }
            ],
            "verdict": {
                "verdict_statement": "Significant improvements achieved.",
                "confidence": "medium",
                "confidence_rationale": "Sufficient paper coverage with abstracts.",
                "key_findings": ["Linear attention reduces computational footprint [doi:10.1000/1]."],
                "limitations_and_gaps": ["Constrained benchmark domains."],
                "top_recommended_papers": ["doi:10.1000/1", "doi:10.1000/2"],
            },
            "synthesis": "Comprehensive overview narrative synthesized across candidate papers.",
        }

        mock_client = MockLLMClient(canned_response=json.dumps(canned_data))
        engine = OverviewEngine(client=mock_client)

        overview = engine.generate_overview(papers=papers, topic="Attention")
        assert len(overview.themes) == 1
        assert overview.themes[0].name == "Foundational Advances"
        assert overview.verdict.confidence == "medium"

    def test_markdown_fenced_json_parsed_cleanly(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [
                {
                    "name": "Theme A",
                    "summary": "Theme summary.",
                    "paper_ids": ["doi:10.1000/1"],
                }
            ],
            "verdict": {
                "verdict_statement": "Verdict text.",
                "confidence": "low",
                "confidence_rationale": "Single paper available.",
                "key_findings": ["Core finding [doi:10.1000/1]."],
                "limitations_and_gaps": ["Single paper evidence."],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis with [doi:10.1000/1].",
        }
        fenced_response = f"```json\n{json.dumps(canned_data)}\n```"
        engine = OverviewEngine(client=MockLLMClient(canned_response=fenced_response))

        overview = engine.generate_overview(papers=papers, topic="Topic")
        assert len(overview.themes) == 1


# ---------------------------------------------------------------------------
# Citation Grounding & Unknown ID Rejection (Requirement 5)
# ---------------------------------------------------------------------------


class TestCitationGroundingAndValidation:
    """Tests verifying that hallucinated, malformed, or index-style citations are strictly rejected."""

    def test_theme_citing_unknown_paper_id_raises_response_error(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [
                {
                    "name": "Hallucinated Theme",
                    "summary": "Summary",
                    "paper_ids": ["doi:10.9999/hallucinated_id"],  # NOT in candidate papers
                }
            ],
            "verdict": {
                "verdict_statement": "Verdict",
                "confidence": "low",
                "confidence_rationale": "Rationale",
                "key_findings": [],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="cited unknown paper_id 'doi:10.9999/hallucinated_id'"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_verdict_top_recs_citing_unknown_paper_id_raises_response_error(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [],
            "verdict": {
                "verdict_statement": "Verdict",
                "confidence": "low",
                "confidence_rationale": "Rationale",
                "key_findings": [],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["arxiv:9999.99999"],  # NOT in candidate papers
            },
            "synthesis": "Synthesis",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="top_recommended_papers cited unknown paper_id"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_finding_citing_unknown_paper_id_raises_response_error(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [],
            "verdict": {
                "verdict_statement": "Verdict",
                "confidence": "low",
                "confidence_rationale": "Rationale",
                "key_findings": ["Significant improvement demonstrated in [doi:10.9999/fake]."],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="Finding cited unknown or unsupported paper_id 'doi:10.9999/fake'"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_synthesis_citing_unknown_paper_id_raises_response_error(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [],
            "verdict": {
                "verdict_statement": "Verdict",
                "confidence": "low",
                "confidence_rationale": "Rationale",
                "key_findings": [],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "As demonstrated in [s2:fake_s2_id], performance degrades.",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="Synthesis cited unknown or unsupported paper_id 's2:fake_s2_id'"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_multiple_citations_in_one_bracket_succeeds(self) -> None:
        papers = [
            _make_paper("doi:10.1000/1", "Paper One"),
            _make_paper("doi:10.1000/2", "Paper Two"),
        ]
        canned_data = {
            "themes": [
                {
                    "name": "Consensus",
                    "summary": "Shared findings.",
                    "paper_ids": ["doi:10.1000/1", "doi:10.1000/2"],
                }
            ],
            "verdict": {
                "verdict_statement": "Verdict.",
                "confidence": "low",
                "confidence_rationale": "Two papers.",
                "key_findings": [
                    "Results confirmed by comma [doi:10.1000/1, doi:10.1000/2] and semicolon [doi:10.1000/1; doi:10.1000/2]."
                ],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis text.",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        overview = engine.generate_overview(papers=papers, topic="Topic")
        assert len(overview.verdict.key_findings) == 1

    def test_multiple_citations_in_one_bracket_with_unknown_raises(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [],
            "verdict": {
                "verdict_statement": "Verdict.",
                "confidence": "low",
                "confidence_rationale": "Rationale.",
                "key_findings": ["Joint finding [doi:10.1000/1, doi:10.9999/unknown]."],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis.",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="cited unknown or unsupported paper_id 'doi:10.9999/unknown'"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_doi_with_parentheses_and_plus_signs_succeeds(self) -> None:
        complex_doi = "doi:10.1002/(sici)1097-0142(19960201)77:3<497::aid-cncr11>3.0.co;2-i"
        plus_doi = "doi:10.1000/1+2"
        papers = [
            _make_paper(complex_doi, "Complex Punctuation Paper"),
            _make_paper(plus_doi, "Plus Sign Paper"),
        ]
        canned_data = {
            "themes": [
                {
                    "name": "Punctuation Support",
                    "summary": "Valid complex DOIs.",
                    "paper_ids": [complex_doi, plus_doi],
                }
            ],
            "verdict": {
                "verdict_statement": "Verdict.",
                "confidence": "low",
                "confidence_rationale": "Two papers.",
                "key_findings": [f"Evidence established in [{complex_doi}, {plus_doi}]."],
                "limitations_and_gaps": [],
                "top_recommended_papers": [complex_doi],
            },
            "synthesis": f"Synthesis referencing [{complex_doi}; {plus_doi}].",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        overview = engine.generate_overview(papers=papers, topic="Topic")
        assert len(overview.themes[0].paper_ids) == 2

    def test_uppercase_doi_normalized_and_preserved_canonically(self) -> None:
        canonical_doi = "doi:10.1000/182+abc"
        papers = [_make_paper(canonical_doi, "Paper One")]
        canned_data = {
            "themes": [
                {
                    "name": "Theme",
                    "summary": "Summary",
                    "paper_ids": ["DOI:10.1000/182+ABC"],  # Uppercase representation
                }
            ],
            "verdict": {
                "verdict_statement": "Verdict.",
                "confidence": "low",
                "confidence_rationale": "Rationale.",
                "key_findings": ["Finding citing [DOI:10.1000/182+ABC]."],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["DOI:10.1000/182+ABC"],
            },
            "synthesis": "Synthesis with [DOI:10.1000/182+ABC].",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        overview = engine.generate_overview(papers=papers, topic="Topic")
        # Canonical ID must be preserved in outputs
        assert overview.themes[0].paper_ids[0] == canonical_doi
        assert overview.verdict.top_recommended_papers[0] == canonical_doi

    @pytest.mark.parametrize("bad_ref", ["[pmid:12345]", "[fake:999]", "[unknown_prefix:42]"])
    def test_unknown_ids_with_arbitrary_prefixes_rejected(self, bad_ref: str) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [],
            "verdict": {
                "verdict_statement": "Verdict.",
                "confidence": "low",
                "confidence_rationale": "Rationale.",
                "key_findings": [f"Unsupported prefix {bad_ref}."],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis.",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="cited unknown or unsupported paper_id"):
            engine.generate_overview(papers=papers, topic="Topic")

    @pytest.mark.parametrize("index_citation", ["[Paper 1]", "[P1]", "[1]", "[1, 2]", "[Paper 1; Paper 2]"])
    def test_index_style_citations_rejected(self, index_citation: str) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [],
            "verdict": {
                "verdict_statement": "Verdict.",
                "confidence": "low",
                "confidence_rationale": "Rationale.",
                "key_findings": [f"Finding with index citation {index_citation}."],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis.",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="invalid index-style citation"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_empty_theme_paper_ids_in_llm_response_raises(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [
                {
                    "name": "Empty Theme",
                    "summary": "No papers linked.",
                    "paper_ids": [],
                }
            ],
            "verdict": {
                "verdict_statement": "Verdict.",
                "confidence": "low",
                "confidence_rationale": "Rationale.",
                "key_findings": [],
                "limitations_and_gaps": [],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Synthesis.",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        with pytest.raises(LLMResponseError, match="at least one paper ID"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_citations_checked_in_each_generated_field(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]

        # Field 1: theme name
        canned_name = {
            "themes": [{"name": "Theme [doi:10.9999/bad]", "summary": "S", "paper_ids": ["doi:10.1000/1"]}],
            "verdict": {"verdict_statement": "V", "confidence": "low", "confidence_rationale": "R"},
            "synthesis": "S",
        }
        with pytest.raises(LLMResponseError, match="Theme #0 name cited unknown or unsupported paper_id"):
            OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_name))).generate_overview(
                papers=papers, topic="Topic"
            )

        # Field 2: theme summary
        canned_summary = {
            "themes": [{"name": "Theme", "summary": "Summary with [Paper 1]", "paper_ids": ["doi:10.1000/1"]}],
            "verdict": {"verdict_statement": "V", "confidence": "low", "confidence_rationale": "R"},
            "synthesis": "S",
        }
        with pytest.raises(LLMResponseError, match="invalid index-style citation"):
            OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_summary))).generate_overview(
                papers=papers, topic="Topic"
            )

        # Field 3: verdict_statement
        canned_vstmt = {
            "themes": [],
            "verdict": {"verdict_statement": "Verdict citing [fake:999]", "confidence": "low", "confidence_rationale": "R"},
            "synthesis": "S",
        }
        with pytest.raises(LLMResponseError, match="Verdict statement cited unknown or unsupported paper_id"):
            OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_vstmt))).generate_overview(
                papers=papers, topic="Topic"
            )

        # Field 4: confidence_rationale
        canned_rationale = {
            "themes": [],
            "verdict": {"verdict_statement": "V", "confidence": "low", "confidence_rationale": "Rationale citing [1]"},
            "synthesis": "S",
        }
        with pytest.raises(LLMResponseError, match="Verdict confidence rationale used invalid index-style citation"):
            OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_rationale))).generate_overview(
                papers=papers, topic="Topic"
            )

        # Field 5: limitations_and_gaps
        canned_lim = {
            "themes": [],
            "verdict": {
                "verdict_statement": "V",
                "confidence": "low",
                "confidence_rationale": "R",
                "limitations_and_gaps": ["Gap noted in [P1]."],
            },
            "synthesis": "S",
        }
        with pytest.raises(LLMResponseError, match="Limitations and gaps used invalid index-style citation"):
            OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_lim))).generate_overview(
                papers=papers, topic="Topic"
            )

    def test_legitimate_prose_brackets_not_treated_as_citations(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned_data = {
            "themes": [
                {
                    "name": "Empirical Scaling [Section 3]",
                    "summary": "Observational analysis [preliminary result].",
                    "paper_ids": ["doi:10.1000/1"],
                }
            ],
            "verdict": {
                "verdict_statement": "Strong trends confirmed [see Fig. 2].",
                "confidence": "low",
                "confidence_rationale": "Single paper evidence [Table 1].",
                "key_findings": ["Loss declines smoothly [doi:10.1000/1] as noted in [Section 3]."],
                "limitations_and_gaps": ["Requires further benchmark diversity [Appendix]."],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Comprehensive narrative [see Fig. 2] supported by [doi:10.1000/1].",
        }
        engine = OverviewEngine(client=MockLLMClient(canned_response=json.dumps(canned_data)))
        overview = engine.generate_overview(papers=papers, topic="Topic")
        assert overview.themes[0].name == "Empirical Scaling [Section 3]"
        assert "see Fig. 2" in overview.verdict.verdict_statement

    def test_extract_bracketed_citations_helper_unit(self) -> None:
        text = (
            "Single [doi:10.1000/1], multi-comma [doi:10.1000/2, doi:10.1000/3], "
            "multi-semicolon [doi:10.1000/4; arxiv:2401.00001], "
            "DOI with internal punctuation [doi:10.1002/(sici)1097-0142...3.0.co;2-i; doi:10.1000/1+2], "
            "index citations [1, 2] and [Paper 1] and [P1], "
            "and legitimate prose [see Fig. 2], [Section 3], [preliminary result], [Appendix]."
        )
        extracted = _extract_bracketed_citations(text)
        assert "doi:10.1000/1" in extracted
        assert "doi:10.1000/2" in extracted
        assert "doi:10.1000/3" in extracted
        assert "doi:10.1000/4" in extracted
        assert "arxiv:2401.00001" in extracted
        assert "doi:10.1002/(sici)1097-0142...3.0.co;2-i" in extracted
        assert "doi:10.1000/1+2" in extracted
        assert "1" in extracted
        assert "2" in extracted
        assert "Paper 1" in extracted
        assert "P1" in extracted
        # Ordinary prose brackets must NOT be extracted as citation tokens
        assert "see Fig. 2" not in extracted
        assert "Section 3" not in extracted
        assert "preliminary result" not in extracted
        assert "Appendix" not in extracted


# ---------------------------------------------------------------------------
# Malformed Responses & Provider Error Handling
# ---------------------------------------------------------------------------


class TestOverviewMalformedResponsesAndErrors:
    """Tests error handling for malformed JSON, provider failures, and invalid structures."""

    def test_malformed_json_raises_llm_response_error(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        engine = OverviewEngine(client=MockLLMClient(canned_response="Not valid JSON."))
        with pytest.raises(LLMResponseError, match="Failed to parse LLM output"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_missing_verdict_field_raises_llm_response_error(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned = json.dumps({"themes": [], "synthesis": "Synthesis"})
        engine = OverviewEngine(client=MockLLMClient(canned_response=canned))
        with pytest.raises(LLMResponseError, match="Field 'verdict' must be a dictionary"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_missing_synthesis_field_raises_llm_response_error(self) -> None:
        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        canned = json.dumps({"themes": [], "verdict": {"verdict_statement": "V", "confidence": "low", "confidence_rationale": "R"}})
        engine = OverviewEngine(client=MockLLMClient(canned_response=canned))
        with pytest.raises(LLMResponseError, match="Field 'synthesis' must be a non-empty string"):
            engine.generate_overview(papers=papers, topic="Topic")

    def test_unsupported_provider_raises_configuration_error(self) -> None:
        with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
            OverviewEngine(provider="unsupported_provider")

    def test_gemini_missing_api_key_raises_configuration_error(self) -> None:
        with (
            patch.object(config, "GEMINI_API_KEY", None),
            pytest.raises(LLMConfigurationError, match="Gemini API key is required"),
        ):
            OverviewEngine(provider="gemini", api_key=None)

    @patch("google.genai.Client")
    def test_gemini_provider_network_error_wrapped(
        self, mock_genai_client_class: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        mock_genai_client_class.return_value = mock_instance
        mock_instance.models.generate_content.side_effect = TimeoutError("Request timed out.")

        client = GeminiClient(api_key="mock_key")
        engine = OverviewEngine(client=client)

        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        with pytest.raises(LLMProviderError, match="Gemini API request failed"):
            engine.generate_overview(papers=papers, topic="Topic")

    @patch("google.genai.Client")
    def test_gemini_successful_overview_generation(
        self, mock_genai_client_class: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        mock_genai_client_class.return_value = mock_instance

        canned = {
            "themes": [
                {
                    "name": "Attention Mechanisms",
                    "summary": "Algorithmic scaling.",
                    "paper_ids": ["doi:10.1000/1"],
                }
            ],
            "verdict": {
                "verdict_statement": "Attention scaling verified.",
                "confidence": "low",
                "confidence_rationale": "Single paper verified.",
                "key_findings": ["Linear attention confirmed [doi:10.1000/1]."],
                "limitations_and_gaps": ["Single paper scope."],
                "top_recommended_papers": ["doi:10.1000/1"],
            },
            "synthesis": "Comprehensive narrative citing [doi:10.1000/1].",
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(canned)
        mock_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="mock_key")
        engine = OverviewEngine(client=client)

        papers = [_make_paper("doi:10.1000/1", "Paper One")]
        overview = engine.generate_overview(papers=papers, topic="Attention")
        assert overview.total_papers_analyzed == 1
        assert overview.verdict.confidence == "low"
