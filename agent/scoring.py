"""Paper scoring and ranking engine — Member B (AI scoring and LLM logic).

Implements the multi-criteria paper scoring algorithm defined in the PRD:
- Citations per year: 40% (normalized across candidate batch)
- Influential-citation ratio: 20%
- Venue quality: 20% (configurable venue tier mapping)
- Recency: 10% (linear decay relative to current_year)
- Peer-review score: 10% (supports multiple scales, redistributed when missing)

Assumptions:
1. Citations per year: Age is computed as `max(1, current_year - paper.year)`. If publication
   year is missing, `default_paper_age` (default 5.0 years) is used if citation count is positive;
   otherwise citation rate is 0.0.
2. Batch normalization: Citation rates are normalized by the maximum rate observed within
   the candidate batch (max scaling). If all candidate papers have zero citation rate, citation
   scores are 0.0.
3. Influential citation ratio: Defined as `influential_citation_count / citation_count`. Clamped to
   [0.0, 1.0] to guard against provider discrepancies. If citation count is None or <= 0, ratio is 0.0.
4. Venue quality: Explicit case-insensitive matching against configurable venue tiers. If venue
   is missing or unrecognized, defaults to `default_venue_score` (0.0).
5. Recency: Linearly decays over `recency_window_years` (default 10 years) from 1.0 (current year)
   to 0.0 (>= 10 years old). Papers with missing year receive 0.0 recency score.
6. Peer-review score: Supports multiple scales (0-1, 1-5, 1-10, 0-100, custom min/max, or auto).
   When missing (None), its 10% weight is redistributed proportionally across the remaining 4 components,
   ensuring fair evaluation without penalizing unreviewed preprints.
7. Non-mutation: Original Paper instances are never mutated; shallow copies are returned with
   `score`, `score_breakdown`, and `is_recommended` populated.
8. Deterministic tie-breaking: Ties on score are broken deterministically by citation count (descending),
   influential citation count (descending), publication year (descending), and canonical paper_id (ascending).

Limitations:
1. Cohort sensitivity: Batch-relative citation normalization means a paper's final score depends
   partially on the distribution of papers in the query cohort.
2. Review score availability: Academic providers rarely index peer review scores; weight redistribution
   is commonly active unless papers are enriched with external peer-review datasets (e.g. OpenReview).
3. Venue alias coverage: Complex journal names or unusual acronyms not in the default dictionary
   fall back to default_venue_score unless supplied via custom venue_scores mapping.
"""

import re
from datetime import datetime, timezone

from shared.models import Paper

# Default base weights summing to 1.0 (100%)
DEFAULT_WEIGHTS: dict[str, float] = {
    "citations_per_year": 0.40,
    "influential_citation_ratio": 0.20,
    "venue_quality": 0.20,
    "recency": 0.10,
    "peer_review": 0.10,
}

# Curated tier-based venue quality dictionary
DEFAULT_VENUE_SCORES: dict[str, float] = {
    # Top-tier AI / ML / NLP / CV / Data (Tier 1: 1.0)
    "nature": 1.0,
    "science": 1.0,
    "cell": 1.0,
    "pnas": 0.95,
    "neurips": 1.0,
    "nips": 1.0,
    "neural information processing systems": 1.0,
    "icml": 1.0,
    "international conference on machine learning": 1.0,
    "iclr": 1.0,
    "international conference on learning representations": 1.0,
    "cvpr": 1.0,
    "computer vision and pattern recognition": 1.0,
    "iccv": 1.0,
    "international conference on computer vision": 1.0,
    "acl": 1.0,
    "association for computational linguistics": 1.0,
    "kdd": 1.0,
    "sigir": 1.0,
    "jmlr": 1.0,
    "journal of machine learning research": 1.0,
    "tpami": 1.0,
    "ieee transactions on pattern analysis and machine intelligence": 1.0,
    "acm computing surveys": 0.95,
    # High-tier AI / Systems / Web (Tier 2: 0.85)
    "aaai": 0.85,
    "association for the advancement of artificial intelligence": 0.85,
    "ijcai": 0.85,
    "international joint conference on artificial intelligence": 0.85,
    "emnlp": 0.85,
    "empirical methods in natural language processing": 0.85,
    "naacl": 0.85,
    "eccv": 0.85,
    "european conference on computer vision": 0.85,
    "www": 0.85,
    "the web conference": 0.85,
    "thewebconf": 0.85,
    "wsdm": 0.85,
    "sigcomm": 0.85,
    "osdi": 0.85,
    "sosp": 0.85,
    "pldi": 0.85,
    "popl": 0.85,
    "focs": 0.85,
    "stoc": 0.85,
    # Solid / Domain-specific conferences and journals (Tier 3: 0.70)
    "coling": 0.70,
    "eacl": 0.70,
    "conll": 0.70,
    "bmvc": 0.70,
    "wacv": 0.70,
    "cikm": 0.70,
    "recsys": 0.70,
    "interspeech": 0.70,
    "icassp": 0.70,
    "bioinformatics": 0.70,
    "ieee access": 0.60,
    "plos one": 0.60,
    # Preprints and Working Papers (Tier 4: 0.40)
    "arxiv": 0.40,
    "biorxiv": 0.40,
    "medrxiv": 0.40,
    "chemrxiv": 0.40,
    "ssrn": 0.40,
}


def normalize_review_score(
    raw_score: float | None,
    scale: str | tuple[float, float] = "auto",
) -> float | None:
    """Normalize a peer-review score to a float bounded strictly between 0.0 and 1.0.

    Parameters:
        raw_score: The raw numerical review score, or None if missing.
        scale: The scale configuration:
            - "auto": dynamically detects [0, 1], [1, 5], [1, 10], or [0, 100].
            - "0-1": standard 0.0 to 1.0 range.
            - "1-5": 1 to 5 scale (e.g. standard conference rating), mapped via (x - 1) / 4.
            - "0-5": 0 to 5 scale, mapped via x / 5.
            - "1-10": 1 to 10 scale (OpenReview/NeurIPS), mapped via (x - 1) / 9.
            - "0-10": 0 to 10 scale, mapped via x / 10.
            - "0-100": percentage score, mapped via x / 100.
            - "1-100": mapped via (x - 1) / 99.
            - tuple[float, float]: custom (min_val, max_val), mapped via (x - min) / (max - min).

    Returns:
        Normalized float in [0.0, 1.0], or None if raw_score is None or invalid.
    """
    if raw_score is None:
        return None

    try:
        val = float(raw_score)
    except (ValueError, TypeError):
        return None

    if isinstance(scale, (tuple, list)) and len(scale) == 2:
        min_v = float(scale[0])
        max_v = float(scale[1])
        if max_v <= min_v:
            return 0.0
        return max(0.0, min(1.0, (val - min_v) / (max_v - min_v)))

    if isinstance(scale, str):
        s = scale.lower().strip()
        if s == "0-1":
            return max(0.0, min(1.0, val))
        elif s == "1-5":
            return max(0.0, min(1.0, (val - 1.0) / 4.0))
        elif s == "0-5":
            return max(0.0, min(1.0, val / 5.0))
        elif s == "1-10":
            return max(0.0, min(1.0, (val - 1.0) / 9.0))
        elif s == "0-10":
            return max(0.0, min(1.0, val / 10.0))
        elif s == "0-100":
            return max(0.0, min(1.0, val / 100.0))
        elif s == "1-100":
            return max(0.0, min(1.0, (val - 1.0) / 99.0))
        elif s == "auto":
            if 0.0 <= val <= 1.0:
                return val
            elif 1.0 < val <= 5.0:
                return max(0.0, min(1.0, (val - 1.0) / 4.0))
            elif 5.0 < val <= 10.0:
                return max(0.0, min(1.0, (val - 1.0) / 9.0))
            elif 10.0 < val <= 100.0:
                return max(0.0, min(1.0, val / 100.0))
            elif val < 0.0:
                return 0.0
            else:
                return 1.0

    return max(0.0, min(1.0, val))


def resolve_venue_quality(
    venue: str | None,
    venue_scores: dict[str, float],
    default_venue_score: float = 0.0,
) -> float:
    """Resolve venue quality score [0.0, 1.0] using explicit and configurable mappings."""
    if not venue or not isinstance(venue, str):
        return max(0.0, min(1.0, default_venue_score))

    clean = venue.strip().lower()
    if not clean:
        return max(0.0, min(1.0, default_venue_score))

    # 1. Exact match in normalized dictionary
    if clean in venue_scores:
        return max(0.0, min(1.0, venue_scores[clean]))

    # 2. Tokenized words and substring matching
    tokens = set(re.findall(r"[a-z0-9]+", clean))
    matches: list[float] = []

    for key, score in venue_scores.items():
        k_clean = key.lower()
        if k_clean in clean or k_clean in tokens:
            matches.append(score)

    if matches:
        return max(0.0, min(1.0, max(matches)))

    return max(0.0, min(1.0, default_venue_score))


def _compute_raw_citation_rate(
    paper: Paper,
    current_year: int,
    default_paper_age: float = 5.0,
) -> float:
    """Calculate raw citations per year."""
    citations = paper.citation_count
    if citations is None or citations <= 0:
        return 0.0

    if paper.year is not None:
        age: float = float(max(1, current_year - paper.year))
    else:
        age = max(1.0, float(default_paper_age))

    return float(citations) / age


def _compute_influential_ratio(paper: Paper) -> float:
    """Calculate ratio of influential citations to total citations [0.0, 1.0]."""
    citations = paper.citation_count
    if citations is None or citations <= 0:
        return 0.0

    influential = paper.influential_citation_count
    if influential is None or influential <= 0:
        return 0.0

    return max(0.0, min(1.0, float(influential) / float(citations)))


def _compute_recency_score(
    paper: Paper,
    current_year: int,
    recency_window_years: int = 10,
) -> float:
    """Calculate recency score [0.0, 1.0] with linear decay."""
    if paper.year is None:
        return 0.0

    age = max(0, current_year - paper.year)
    if age >= recency_window_years:
        return 0.0

    score = 1.0 - (float(age) / float(recency_window_years))
    return max(0.0, min(1.0, score))


def score_papers(
    papers: list[Paper],
    threshold: float = 0.60,
    current_year: int | None = None,
    *,
    venue_scores: dict[str, float] | None = None,
    default_venue_score: float = 0.0,
    weights: dict[str, float] | None = None,
    review_scale: str | tuple[float, float] = "auto",
    recency_window_years: int = 10,
    default_paper_age: float = 5.0,
) -> list[Paper]:
    """Score, rank, and recommend research papers using a multi-criteria model.

    Formula (PRD):
    - Citations per year: 40% (batch-normalized)
    - Influential-citation ratio: 20%
    - Venue quality: 20%
    - Recency: 10%
    - Peer-review score: 10%

    If a paper lacks a peer-review score, its 10% weight is redistributed proportionally
    across the remaining 4 components.

    Parameters:
        papers: List of shared Paper objects to score.
        threshold: Score threshold for recommendation (is_recommended = score >= threshold).
        current_year: Reference year for age and recency calculations (injectable for testing).
        venue_scores: Optional dictionary of venue name -> quality score overrides.
        default_venue_score: Score to assign unknown or missing venues. Default 0.0.
        weights: Optional dictionary of component weights (must include component keys).
        review_scale: Scale for normalizing review scores ("auto", "0-1", "1-5", "1-10", "0-100", etc.).
        recency_window_years: Number of years over which recency score decays to 0.0.
        default_paper_age: Fallback age in years if publication year is missing but citations exist.

    Returns:
        New list of Paper objects sorted by descending score with `score`, `score_breakdown`,
        and `is_recommended` populated. Original Paper objects are preserved unmutated.
    """
    if not papers:
        return []

    # 1. Determine reference current year
    curr_year: int = current_year if current_year is not None else datetime.now(timezone.utc).year

    # 2. Build merged venue score dictionary
    effective_venue_scores: dict[str, float] = {**DEFAULT_VENUE_SCORES}
    if venue_scores:
        for k, v in venue_scores.items():
            effective_venue_scores[k.strip().lower()] = max(0.0, min(1.0, float(v)))

    # 3. Base weights configuration
    w_cfg = weights if weights is not None else DEFAULT_WEIGHTS
    w_cit = float(w_cfg.get("citations_per_year", 0.40))
    w_inf = float(w_cfg.get("influential_citation_ratio", 0.20))
    w_ven = float(w_cfg.get("venue_quality", 0.20))
    w_rec = float(w_cfg.get("recency", 0.10))
    w_rev = float(w_cfg.get("peer_review", 0.10))

    # 4. Batch citation rates calculation and normalization
    raw_citation_rates: list[float] = [
        _compute_raw_citation_rate(p, curr_year, default_paper_age)
        for p in papers
    ]
    max_rate: float = max(raw_citation_rates) if raw_citation_rates else 0.0

    scored_entries: list[Paper] = []

    for i, paper in enumerate(papers):
        # Component 1: Citations per year (batch normalized)
        raw_r = raw_citation_rates[i]
        c_score = min(1.0, max(0.0, raw_r / max_rate)) if max_rate > 0.0 else 0.0

        # Component 2: Influential-citation ratio
        inf_score = _compute_influential_ratio(paper)

        # Component 3: Venue quality
        ven_score = resolve_venue_quality(paper.venue, effective_venue_scores, default_venue_score)

        # Component 4: Recency
        rec_score = _compute_recency_score(paper, curr_year, recency_window_years)

        # Component 5: Peer-review score & weight redistribution
        rev_norm = normalize_review_score(paper.review_score, scale=review_scale)

        if rev_norm is not None:
            # All 5 components are present
            total_weight = w_cit + w_inf + w_ven + w_rec + w_rev
            if total_weight > 0.0:
                raw_final = (
                    w_cit * c_score
                    + w_inf * inf_score
                    + w_ven * ven_score
                    + w_rec * rec_score
                    + w_rev * rev_norm
                ) / total_weight
            else:
                raw_final = 0.0
            breakdown_rev_score = rev_norm
        else:
            # Peer review score missing: redistribute w_rev proportionally among remaining 4
            rem_weight = w_cit + w_inf + w_ven + w_rec
            if rem_weight > 0.0:
                raw_final = (
                    w_cit * c_score
                    + w_inf * inf_score
                    + w_ven * ven_score
                    + w_rec * rec_score
                ) / rem_weight
            else:
                raw_final = 0.0
            breakdown_rev_score = 0.0

        final_score = round(max(0.0, min(1.0, raw_final)), 4)
        is_rec = bool(final_score >= threshold)

        breakdown = {
            "citations_per_year": round(c_score, 4),
            "influential_citation_ratio": round(inf_score, 4),
            "venue_quality": round(ven_score, 4),
            "recency": round(rec_score, 4),
            "peer_review": round(breakdown_rev_score, 4),
        }

        # Clone paper preserving all fields and avoiding mutation of input object
        scored_paper = paper.model_copy(
            update={
                "score": final_score,
                "score_breakdown": breakdown,
                "is_recommended": is_rec,
            }
        )
        scored_entries.append(scored_paper)

    # 5. Deterministic sorting:
    # Primary: score (descending)
    # Tie-breakers: citation_count (descending), influential_citations (descending),
    # year (descending), paper_id (ascending string)
    def tie_break_key(p: Paper) -> tuple[float, int, int, int, str]:
        s = p.score if p.score is not None else 0.0
        c = p.citation_count if p.citation_count is not None else 0
        inf = p.influential_citation_count if p.influential_citation_count is not None else 0
        y = p.year if p.year is not None else 0
        pid = str(p.paper_id)
        return (-s, -c, -inf, -y, pid)

    scored_entries.sort(key=tie_break_key)
    return scored_entries
