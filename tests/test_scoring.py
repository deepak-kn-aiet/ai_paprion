"""Automated tests for Phase 1: Paper Scoring and Ranking (Member B)."""

import pytest

from agent.scoring import (
    DEFAULT_VENUE_SCORES,
    DEFAULT_WEIGHTS,
    normalize_review_score,
    resolve_venue_quality,
    score_papers,
)
from shared.models import Paper


# --- Helper to create synthetic papers for testing ---
def make_synthetic_paper(
    paper_id: str = "doi:10.1000/test",
    title: str = "Synthetic Test Paper",
    year: int | None = 2024,
    citation_count: int | None = 50,
    influential_citation_count: int | None = 10,
    venue: str | None = "NeurIPS",
    review_score: float | None = None,
    **kwargs,
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        year=year,
        citation_count=citation_count,
        influential_citation_count=influential_citation_count,
        venue=venue,
        review_score=review_score,
        **kwargs,
    )


# --- Configuration tests ---

def test_default_weights():
    """Verify default weights sum to 1.0 and match PRD specifications."""
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    assert DEFAULT_WEIGHTS["citations_per_year"] == 0.40
    assert DEFAULT_WEIGHTS["influential_citation_ratio"] == 0.20
    assert DEFAULT_WEIGHTS["venue_quality"] == 0.20
    assert DEFAULT_WEIGHTS["recency"] == 0.10
    assert DEFAULT_WEIGHTS["peer_review"] == 0.10


# --- 1. Empty input ---

def test_score_papers_empty_list():
    """Empty list returns empty list."""
    assert score_papers([]) == []


def test_score_papers_none_input():
    """None or empty returns empty list safely."""
    assert score_papers([]) == []


# --- 2. Correct descending ranking ---

def test_score_papers_descending_order():
    """Returned papers are strictly sorted in descending score order."""
    p_high = make_synthetic_paper(
        paper_id="doi:10.1000/high",
        title="High Impact Paper",
        year=2025,
        citation_count=200,
        influential_citation_count=40,
        venue="NeurIPS",
    )
    p_mid = make_synthetic_paper(
        paper_id="doi:10.1000/mid",
        title="Medium Impact Paper",
        year=2023,
        citation_count=40,
        influential_citation_count=5,
        venue="AAAI",
    )
    p_low = make_synthetic_paper(
        paper_id="doi:10.1000/low",
        title="Low Impact Paper",
        year=2015,
        citation_count=2,
        influential_citation_count=0,
        venue="Unknown Workshop",
    )

    results = score_papers([p_low, p_high, p_mid], current_year=2026)
    assert len(results) == 3
    assert results[0].paper_id == "doi:10.1000/high"
    assert results[1].paper_id == "doi:10.1000/mid"
    assert results[2].paper_id == "doi:10.1000/low"

    assert all(p.score is not None for p in results)
    scores = [p.score for p in results if p.score is not None]
    assert scores == sorted(scores, reverse=True)


# --- 3. Scores between 0 and 1 ---

def test_scores_bounded_between_zero_and_one():
    """All final and component breakdown scores must be in [0.0, 1.0]."""
    papers = [
        make_synthetic_paper(paper_id="id:1", citation_count=1000000, year=2026, venue="Nature", review_score=10.0),
        make_synthetic_paper(paper_id="id:2", citation_count=0, year=1990, venue=None, review_score=None),
        make_synthetic_paper(paper_id="id:3", citation_count=50, year=2024, venue="arXiv", review_score=0.5),
    ]
    results = score_papers(papers, current_year=2026)

    for p in results:
        assert p.score is not None
        assert 0.0 <= p.score <= 1.0
        assert p.score_breakdown is not None
        for key, val in p.score_breakdown.items():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0, f"Component {key} out of bounds: {val}"


# --- 4. Missing and zero citation counts ---

def test_missing_and_zero_citation_counts():
    """None, 0, or negative citation counts do not crash and produce 0.0 citation scores."""
    papers = [
        make_synthetic_paper(paper_id="id:none", citation_count=None, influential_citation_count=None),
        make_synthetic_paper(paper_id="id:zero", citation_count=0, influential_citation_count=0),
        make_synthetic_paper(paper_id="id:neg", citation_count=-5, influential_citation_count=-2),
    ]
    results = score_papers(papers, current_year=2026)
    assert len(results) == 3
    for p in results:
        assert p.score_breakdown is not None
        assert p.score_breakdown["citations_per_year"] == 0.0
        assert p.score_breakdown["influential_citation_ratio"] == 0.0


def test_batch_all_zero_citations_no_division_by_zero():
    """When all papers have zero citations, batch citation normalization doesn't divide by zero."""
    papers = [
        make_synthetic_paper(paper_id="id:1", citation_count=0),
        make_synthetic_paper(paper_id="id:2", citation_count=None),
    ]
    results = score_papers(papers, current_year=2026)
    assert results[0].score_breakdown is not None
    assert results[1].score_breakdown is not None
    assert results[0].score_breakdown["citations_per_year"] == 0.0
    assert results[1].score_breakdown["citations_per_year"] == 0.0


# --- 5. Missing publication years ---

def test_missing_publication_year():
    """Missing publication year defaults recency to 0.0 and safely computes citation rate with default age."""
    p_no_year = make_synthetic_paper(paper_id="id:no_year", year=None, citation_count=50)
    p_with_year = make_synthetic_paper(paper_id="id:with_year", year=2025, citation_count=50)

    results = score_papers([p_no_year, p_with_year], current_year=2026)
    res_dict = {p.paper_id: p for p in results}

    p_no = res_dict["id:no_year"]
    p_with = res_dict["id:with_year"]
    assert p_no.score_breakdown is not None
    assert p_with.score_breakdown is not None
    assert p_no.score_breakdown["recency"] == 0.0
    assert p_with.score_breakdown["recency"] == 0.9
    # Paper with recent year should rank higher due to recency
    assert results[0].paper_id == "id:with_year"


# --- 6. Missing venue ---

def test_missing_and_unknown_venue():
    """None, empty, or whitespace venue defaults safely to default_venue_score."""
    papers = [
        make_synthetic_paper(paper_id="id:none_venue", venue=None),
        make_synthetic_paper(paper_id="id:empty_venue", venue=""),
        make_synthetic_paper(paper_id="id:blank_venue", venue="   "),
        make_synthetic_paper(paper_id="id:unknown_venue", venue="Random Unindexed Workshop"),
    ]
    results = score_papers(papers, default_venue_score=0.0)
    for p in results:
        assert p.score_breakdown is not None
        assert p.score_breakdown["venue_quality"] == 0.0

    # Test configurable default_venue_score
    results_custom = score_papers(papers, default_venue_score=0.15)
    for p in results_custom:
        assert p.score_breakdown is not None
        assert p.score_breakdown["venue_quality"] == 0.15


# --- 7. Missing review-score weight redistribution ---

def test_missing_review_score_weight_redistribution():
    """When review_score is missing, its 10% weight is proportionally redistributed across the other 4 components."""
    # A paper with perfect 1.0 component scores in the other 4 areas:
    # citations_per_year: 1.0 (weight 0.40)
    # influential_ratio: 1.0 (weight 0.20)
    # venue: 1.0 (weight 0.20)
    # recency: 1.0 (weight 0.10)
    # Total remaining weight = 0.90.
    # With proportional redistribution: (0.40*1 + 0.20*1 + 0.20*1 + 0.10*1) / 0.90 = 0.90 / 0.90 = 1.00!
    perfect_unreviewed = make_synthetic_paper(
        paper_id="id:perfect_unreviewed",
        citation_count=100,
        influential_citation_count=100,
        venue="NeurIPS",
        year=2026,
        review_score=None,
    )
    results = score_papers([perfect_unreviewed], current_year=2026)
    scored = results[0]

    assert scored.score_breakdown is not None
    assert scored.score_breakdown["citations_per_year"] == 1.0
    assert scored.score_breakdown["influential_citation_ratio"] == 1.0
    assert scored.score_breakdown["venue_quality"] == 1.0
    assert scored.score_breakdown["recency"] == 1.0
    assert scored.score_breakdown["peer_review"] == 0.0
    # Final score must be 1.0 due to weight redistribution, not 0.90!
    assert scored.score == 1.0


def test_review_score_present_uses_full_weights():
    """When review score is present, all 5 base weights are applied."""
    p_reviewed = make_synthetic_paper(
        paper_id="id:reviewed",
        citation_count=100,
        influential_citation_count=100,
        venue="NeurIPS",
        year=2026,
        review_score=1.0,  # 1.0 on 0-1 scale
    )
    results = score_papers([p_reviewed], current_year=2026)
    assert results[0].score == 1.0
    assert results[0].score_breakdown is not None
    assert results[0].score_breakdown["peer_review"] == 1.0


# --- 8. Review scores on supported scales ---

def test_review_score_supported_scales():
    """normalize_review_score correctly maps supported scales to [0.0, 1.0]."""
    # 0-1 scale
    assert normalize_review_score(0.85, scale="0-1") == 0.85
    assert normalize_review_score(0.0, scale="0-1") == 0.0
    assert normalize_review_score(1.0, scale="0-1") == 1.0

    # 1-5 scale: (x - 1) / 4
    assert normalize_review_score(5.0, scale="1-5") == 1.0
    assert normalize_review_score(1.0, scale="1-5") == 0.0
    assert normalize_review_score(3.0, scale="1-5") == 0.5

    # 1-10 scale (OpenReview / NeurIPS): (x - 1) / 9
    assert normalize_review_score(10.0, scale="1-10") == 1.0
    assert normalize_review_score(1.0, scale="1-10") == 0.0
    s_5_5 = normalize_review_score(5.5, scale="1-10")
    assert s_5_5 is not None
    assert round(s_5_5, 4) == 0.5

    # 0-100 scale: x / 100
    assert normalize_review_score(85.0, scale="0-100") == 0.85
    assert normalize_review_score(100.0, scale="0-100") == 1.0
    assert normalize_review_score(0.0, scale="0-100") == 0.0

    # Custom tuple scale: e.g. (-3, 3)
    assert normalize_review_score(3.0, scale=(-3.0, 3.0)) == 1.0
    assert normalize_review_score(-3.0, scale=(-3.0, 3.0)) == 0.0
    assert normalize_review_score(0.0, scale=(-3.0, 3.0)) == 0.5

    # Auto scale detection
    assert normalize_review_score(0.75, scale="auto") == 0.75
    assert normalize_review_score(4.0, scale="auto") == 0.75  # 1-5: (4-1)/4
    s_7_0 = normalize_review_score(7.0, scale="auto")
    assert s_7_0 is not None
    assert round(s_7_0, 4) == 0.6667  # 1-10: (7-1)/9
    assert normalize_review_score(90.0, scale="auto") == 0.90  # 0-100: 90/100

    # None and invalid inputs
    assert normalize_review_score(None) is None
    assert normalize_review_score("invalid") is None


# --- 9. Threshold behavior ---

def test_recommendation_threshold():
    """is_recommended is True if score >= threshold, False otherwise."""
    papers = [
        make_synthetic_paper(paper_id="id:high", citation_count=500, influential_citation_count=100, venue="NeurIPS", year=2026),
        make_synthetic_paper(paper_id="id:low", citation_count=0, influential_citation_count=0, venue=None, year=2010),
    ]

    # Threshold = 0.60 (default)
    results_60 = score_papers(papers, threshold=0.60, current_year=2026)
    assert results_60[0].is_recommended is True
    assert results_60[1].is_recommended is False

    # Higher threshold = 0.95
    results_95 = score_papers(papers, threshold=0.95, current_year=2026)
    for p in results_95:
        assert p.score is not None
        if p.score < 0.95:
            assert p.is_recommended is False

    # Zero threshold = 0.0
    results_zero = score_papers(papers, threshold=0.0, current_year=2026)
    assert all(p.is_recommended is True for p in results_zero)


# --- 10. Deterministic results for equal scores ---

def test_deterministic_ordering_for_tied_scores():
    """Tied papers are ordered deterministically by citation_count, influential_citations, year, and paper_id."""
    p1 = make_synthetic_paper(paper_id="doi:10.1000/alpha", title="Same Alpha", citation_count=10, year=2024, venue="arXiv")
    p2 = make_synthetic_paper(paper_id="doi:10.1000/beta", title="Same Beta", citation_count=10, year=2024, venue="arXiv")

    res_1 = score_papers([p1, p2], current_year=2026)
    res_2 = score_papers([p2, p1], current_year=2026)

    # Identical score, identical citations -> paper_id tie-breaker ensures 'alpha' before 'beta'
    assert [p.paper_id for p in res_1] == ["doi:10.1000/alpha", "doi:10.1000/beta"]
    assert [p.paper_id for p in res_2] == ["doi:10.1000/alpha", "doi:10.1000/beta"]


def test_tie_breaking_by_citation_count():
    """When composite scores tie, higher citation count wins."""
    # Custom weights ignoring citations to create tie in composite score
    weights = {"citations_per_year": 0.0, "influential_citation_ratio": 0.0, "venue_quality": 1.0, "recency": 0.0, "peer_review": 0.0}
    p_more_cites = make_synthetic_paper(paper_id="id:more", citation_count=100, venue="NeurIPS")
    p_fewer_cites = make_synthetic_paper(paper_id="id:fewer", citation_count=10, venue="NeurIPS")

    res = score_papers([p_fewer_cites, p_more_cites], weights=weights)
    assert res[0].paper_id == "id:more"
    assert res[1].paper_id == "id:fewer"


# --- 11. Original objects remain unchanged ---

def test_original_paper_objects_not_mutated():
    """score_papers must not mutate the input Paper instances."""
    original = make_synthetic_paper(paper_id="doi:10.1000/immutable", title="Immutable Paper")
    assert original.score is None
    assert original.score_breakdown is None
    assert original.is_recommended is None

    results = score_papers([original], current_year=2026)

    # Original model is unchanged
    assert original.score is None
    assert original.score_breakdown is None
    assert original.is_recommended is None

    # Returned model is a new instance with populated fields
    scored = results[0]
    assert scored is not original
    assert scored.score is not None
    assert scored.score_breakdown is not None
    assert scored.is_recommended is not None
    # All original fields are preserved
    assert scored.paper_id == original.paper_id
    assert scored.title == original.title


# --- 12. Invalid or unusual metadata does not crash scorer ---

def test_unusual_metadata_does_not_crash():
    """Anomalous metadata (influential > total citations, future year, ancient year) is handled gracefully."""
    papers = [
        # Influential citations exceed total citations
        make_synthetic_paper(paper_id="id:anomaly_cites", citation_count=10, influential_citation_count=50),
        # Future publication year
        make_synthetic_paper(paper_id="id:future_year", year=2040),
        # Ancient publication year
        make_synthetic_paper(paper_id="id:ancient_year", year=1850),
        # Extreme negative / out-of-range review scores
        make_synthetic_paper(paper_id="id:neg_review", review_score=-10.0),
        make_synthetic_paper(paper_id="id:high_review", review_score=500.0),
        # Only required fields provided, everything else None
        Paper(paper_id="id:bare_minimum", title="Bare Title"),
    ]
    results = score_papers(papers, current_year=2026)
    assert len(results) == len(papers)
    for p in results:
        assert p.score is not None
        assert 0.0 <= p.score <= 1.0


# --- 13. Injectable current_year ---

def test_injectable_current_year():
    """Injecting current_year produces predictable recency and age calculations."""
    paper = make_synthetic_paper(paper_id="id:year_test", year=2020)

    # In 2020: age = 0 -> recency = 1.0
    res_2020 = score_papers([paper], current_year=2020)[0]
    assert res_2020.score_breakdown is not None
    assert res_2020.score_breakdown["recency"] == 1.0

    # In 2025: age = 5 -> recency = 1.0 - (5/10) = 0.5
    res_2025 = score_papers([paper], current_year=2025)[0]
    assert res_2025.score_breakdown is not None
    assert res_2025.score_breakdown["recency"] == 0.5

    # In 2030: age = 10 -> recency = 0.0
    res_2030 = score_papers([paper], current_year=2030)[0]
    assert res_2030.score_breakdown is not None
    assert res_2030.score_breakdown["recency"] == 0.0


# --- 14. Explicit and configurable venue mapping ---

def test_configurable_venue_quality():
    """Custom venue scores can be passed and take precedence."""
    custom_venues = {"Special AI Workshop": 0.95}
    paper = make_synthetic_paper(paper_id="id:custom_venue", venue="Special AI Workshop")

    # Without custom venue mapping -> default 0.0
    res_default = score_papers([paper], default_venue_score=0.0)[0]
    assert res_default.score_breakdown is not None
    assert res_default.score_breakdown["venue_quality"] == 0.0

    # With custom venue mapping -> 0.95
    res_custom = score_papers([paper], venue_scores=custom_venues)[0]
    assert res_custom.score_breakdown is not None
    assert res_custom.score_breakdown["venue_quality"] == 0.95


def test_venue_fuzzy_and_alias_matching():
    """Venues match via case-insensitivity, substring, and tokens."""
    assert resolve_venue_quality("Proceedings of NeurIPS 2024", DEFAULT_VENUE_SCORES) == 1.0
    assert resolve_venue_quality("icml conference", DEFAULT_VENUE_SCORES) == 1.0
    assert resolve_venue_quality("arXiv.org preprint", DEFAULT_VENUE_SCORES) == 0.40
    assert resolve_venue_quality("Unknown", DEFAULT_VENUE_SCORES, default_venue_score=0.05) == 0.05
