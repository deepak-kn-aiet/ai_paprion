"""Executive overview, evidence-grounded synthesis, and research verdict components."""

from __future__ import annotations

import html
import re

import streamlit as st

from agent.agent import ResearchAgentResult
from agent.overview import ResearchOverview
from shared.models import Paper


def get_paper_external_url(paper_id: str, paper: Paper | None = None) -> str | None:
    """Resolve an external URL for a paper reference (DOI, arXiv, or direct URL)."""
    if paper:
        if paper.url and paper.url.startswith(("http://", "https://")):
            return paper.url
        pdf_url = getattr(paper, "pdf_url", None) or getattr(paper, "open_access_pdf_url", None)
        if pdf_url and pdf_url.startswith(("http://", "https://")):
            return pdf_url
        if paper.doi:
            doi_clean = paper.doi.strip()
            if doi_clean.startswith("http"):
                return doi_clean
            return f"https://doi.org/{doi_clean.lstrip('/')}"

    clean_id = (paper_id or "").strip()
    if clean_id.startswith(("http://", "https://")):
        return clean_id
    if clean_id.lower().startswith("doi:"):
        doi_suffix = clean_id[4:].strip()
        return f"https://doi.org/{doi_suffix.lstrip('/')}"
    if clean_id.lower().startswith("arxiv:"):
        arxiv_suffix = clean_id[6:].strip()
        return f"https://arxiv.org/abs/{arxiv_suffix.lstrip('/')}"
    return None


def extract_citation_tokens(text: str) -> list[str]:
    """Extract canonical citation tokens like [doi:...] or [arxiv:...] from text."""
    if not text or not isinstance(text, str):
        return []
    pattern = r"\[([a-zA-Z0-9_\-\.\:\/]+)\]"
    matches = re.findall(pattern, text)
    tokens: list[str] = []
    for m in matches:
        m_clean = m.strip()
        if m_clean and m_clean not in tokens:
            tokens.append(m_clean)
    return tokens


def format_authors_short(authors: list[str]) -> str:
    """Format author list into a concise academic attribution snippet."""
    if not authors:
        return "Unknown Authors"
    cleaned = [a.strip() for a in authors if isinstance(a, str) and a.strip()]
    if not cleaned:
        return "Unknown Authors"
    if len(cleaned) == 1:
        return html.escape(cleaned[0])
    if len(cleaned) == 2:
        return f"{html.escape(cleaned[0])} &amp; {html.escape(cleaned[1])}"
    return f"{html.escape(cleaned[0])} et al."


def format_finding_citations(text: str, paper_lookup: dict[str, Paper] | None = None) -> str:
    """Format inline citations within a finding or limitation statement."""
    escaped = html.escape(text or "")
    pattern = r"\[([a-zA-Z0-9_\-\.\:\/]+)\]"

    def _replace_citation(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        paper = paper_lookup.get(token.lower()) if paper_lookup else None
        url = get_paper_external_url(token, paper)
        tooltip = ""
        if paper and paper.title:
            tooltip = f"{paper.title} ({paper.year})" if paper.year else paper.title
        elif url:
            tooltip = f"External source: {token}"

        title_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""
        if url:
            return (
                f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer" '
                f'class="inline-citation verdict-inline-citation"{title_attr}>[{html.escape(token)}]</a>'
            )
        return f'<span class="inline-citation verdict-inline-citation"{title_attr}>[{html.escape(token)}]</span>'

    return re.sub(pattern, _replace_citation, escaped)


def format_synthesis_citations(text: str, paper_lookup: dict[str, Paper] | None = None) -> str:
    """Format narrative synthesis citations into styled editorial badge elements."""
    escaped = html.escape(text or "")
    pattern = r"\[([a-zA-Z0-9_\-\.\:\/]+)\]"

    def _replace_citation(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        paper = paper_lookup.get(token.lower()) if paper_lookup else None
        url = get_paper_external_url(token, paper)
        tooltip = ""
        if paper and paper.title:
            tooltip = f"{paper.title} ({paper.year})" if paper.year else paper.title
        elif url:
            tooltip = f"External source: {token}"

        title_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""
        if url:
            return (
                f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer" '
                f'class="inline-citation verdict-inline-citation"{title_attr}>[{html.escape(token)}]</a>'
            )
        return f'<span class="inline-citation verdict-inline-citation"{title_attr}>[{html.escape(token)}]</span>'

    styled = re.sub(pattern, _replace_citation, escaped)
    paragraphs = [p.strip() for p in styled.split("\n\n") if p.strip()]
    if paragraphs:
        return "".join(f'<p class="synthesis-paragraph">{p}</p>' for p in paragraphs)
    return f'<p class="synthesis-paragraph">{styled}</p>'


def render_verdict_statement_box(overview: ResearchOverview) -> None:
    """Render executive verdict statement, confidence badge, and confidence rationale."""
    conf_str = (overview.verdict.confidence or "").strip()
    if conf_str:
        conf_class = f"verdict-confidence-{conf_str.lower()}"
        conf_badge = f'<span class="verdict-confidence-badge {conf_class}">CONFIDENCE: {html.escape(conf_str.upper())}</span>'
    else:
        conf_badge = ""

    topic_html = ""
    if overview.topic:
        topic_html = (
            f'<div class="verdict-topic-badge">'
            f'<span class="verdict-topic-label">RESEARCH QUESTION</span>'
            f'<span class="verdict-topic-text">{html.escape(overview.topic)}</span>'
            f'</div>'
        )

    corpus_details = f"{overview.total_papers_analyzed} Papers Analyzed"
    if overview.papers_with_abstracts > 0:
        corpus_details += f" ({overview.papers_with_abstracts} with Abstracts)"
    corpus_badge = f'<span class="verdict-corpus-badge">{corpus_details}</span>'

    rationale_html = ""
    if overview.verdict.confidence_rationale:
        rationale_html = (
            f'<div class="verdict-rationale">'
            f'<strong>Confidence Justification:</strong> {html.escape(overview.verdict.confidence_rationale)}'
            f'</div>'
        )

    timings_html = ""
    if overview.timings:
        chips: list[str] = []
        if "evidence_packet" in overview.timings:
            chips.append(f'<span class="verdict-timing-chip">Packet: {overview.timings["evidence_packet"]:.2f}s</span>')
        if "synthesis" in overview.timings:
            chips.append(f'<span class="verdict-timing-chip">Synthesis: {overview.timings["synthesis"]:.2f}s</span>')
        if "citation_validation" in overview.timings:
            chips.append(f'<span class="verdict-timing-chip">Validation: {overview.timings["citation_validation"]:.2f}s</span>')
        if chips:
            timings_html = f'<div class="verdict-timings-strip">{"".join(chips)}</div>'

    verdict_stmt = html.escape(overview.verdict.verdict_statement or "No definitive verdict statement available.")

    st.markdown(
        f"""
        <div class="verdict-box">
            <div class="verdict-header-row">
                <div class="verdict-meta-left">
                    {topic_html}
                    {corpus_badge}
                </div>
                {conf_badge}
            </div>
            <div class="verdict-box-statement">{verdict_stmt}</div>
            {rationale_html}
            {timings_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_overview(overview: ResearchOverview, result: ResearchAgentResult) -> None:
    """Render executive verdict, narrative synthesis, and research themes with editorial styling."""
    st.markdown('<div id="overview-section"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Executive Research Overview")
        render_verdict_statement_box(overview)

        # Build paper lookup for synthesis citations
        paper_lookup: dict[str, Paper] = {}
        for p in result.ranked_papers:
            if p.paper_id:
                paper_lookup[p.paper_id] = p
                paper_lookup[p.paper_id.lower()] = p
            if p.doi:
                doi_clean = p.doi.strip().lower()
                paper_lookup[doi_clean] = p
                paper_lookup[f"doi:{doi_clean}"] = p

        # Narrative Synthesis
        if overview.synthesis:
            st.markdown("##### Evidence-Grounded Synthesis")
            st.markdown(
                f'<div class="synthesis-box">{format_synthesis_citations(overview.synthesis, paper_lookup)}</div>',
                unsafe_allow_html=True,
            )

        # Thematic Clusters
        if overview.themes:
            with st.expander(f"Research Themes ({len(overview.themes)})", expanded=True):
                theme_cols = st.columns(2) if len(overview.themes) > 1 else [st.container()]
                for idx, theme in enumerate(overview.themes):
                    col = theme_cols[idx % len(theme_cols)]
                    with col:
                        resolved_items: list[str] = []
                        for pid in theme.paper_ids[:3]:
                            p = paper_lookup.get(pid.lower())
                            if p:
                                venue_info = f" ({p.venue} {p.year})" if p.venue and p.year else (f" ({p.year})" if p.year else "")
                                title_snippet = p.title[:55] + ("..." if len(p.title) > 55 else "")
                                resolved_items.append(f"<li>{html.escape(title_snippet)}{venue_info}</li>")
                            else:
                                resolved_items.append(f"<li><code>{html.escape(pid)}</code></li>")

                        remaining = len(theme.paper_ids) - 3
                        if remaining > 0:
                            resolved_items.append(f"<li><em>+{remaining} additional papers</em></li>")

                        papers_html = f"<ul class='theme-paper-list'>{''.join(resolved_items)}</ul>" if resolved_items else ""

                        st.markdown(
                            f"""
                            <div class="theme-card">
                                <div class="theme-card-header">
                                    <span class="theme-card-num">Theme #{idx + 1}</span>
                                    <span class="theme-card-title">{html.escape(theme.name)}</span>
                                </div>
                                <div class="theme-card-summary">{html.escape(theme.summary)}</div>
                                <div class="theme-card-sub">Contributing papers ({len(theme.paper_ids)}):</div>
                                {papers_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


def render_verdict_section(overview: ResearchOverview, papers: list[Paper] | None = None) -> None:
    """Render findings, limitations, literature grounding, and verdict section in Light Editorial style."""
    st.markdown('<div id="verdict-section"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Research Verdict & Findings")
        render_verdict_statement_box(overview)

        # Build comprehensive paper lookup
        paper_lookup: dict[str, Paper] = {}
        if papers:
            for p in papers:
                if p.paper_id:
                    paper_lookup[p.paper_id] = p
                    paper_lookup[p.paper_id.lower()] = p
                if p.doi:
                    doi_clean = p.doi.strip().lower()
                    paper_lookup[doi_clean] = p
                    paper_lookup[f"doi:{doi_clean}"] = p
                    if doi_clean.startswith("https://doi.org/"):
                        paper_lookup[doi_clean.removeprefix("https://doi.org/")] = p

        # Key Findings
        findings = overview.verdict.key_findings or []
        st.markdown(f"##### Key Findings ({len(findings)})")
        if not findings:
            st.markdown(
                '<div class="verdict-empty-notice">No verified key factual findings were extracted from the candidate literature.</div>',
                unsafe_allow_html=True,
            )
        else:
            for idx, finding in enumerate(findings, 1):
                cited_tokens = extract_citation_tokens(finding)
                statement_html = format_finding_citations(finding, paper_lookup)

                # Collect matched papers for evidence grounding
                matched_papers: list[tuple[str, Paper]] = []
                seen_pids: set[str] = set()
                for token in cited_tokens:
                    p = paper_lookup.get(token.lower())
                    if p and p.paper_id not in seen_pids:
                        seen_pids.add(p.paper_id)
                        matched_papers.append((token, p))

                evidence_html = ""
                if matched_papers:
                    items_html: list[str] = []
                    for token, p in matched_papers:
                        p_url = get_paper_external_url(token, p)
                        title_esc = html.escape(p.title)
                        if p_url:
                            p_link = f'<a href="{html.escape(p_url)}" target="_blank" rel="noopener noreferrer" class="finding-evidence-paper-title">{title_esc}</a>'
                        else:
                            p_link = f'<span class="finding-evidence-paper-title">{title_esc}</span>'

                        authors_str = format_authors_short(p.authors)
                        venue_year = f"{html.escape(p.venue)} ({p.year})" if p.venue and p.year else (f"({p.year})" if p.year else (html.escape(p.venue) if p.venue else ""))
                        citations_str = f" · {p.citation_count:,} citations" if p.citation_count is not None else ""

                        items_html.append(
                            f'<div class="finding-evidence-item">'
                            f'<span class="finding-evidence-bullet">▸</span>'
                            f'<div class="finding-evidence-details">{p_link}<span class="finding-evidence-meta">{authors_str} {venue_year}{citations_str}</span></div>'
                            f'</div>'
                        )
                    evidence_html = (
                        f'<div class="finding-evidence-box">'
                        f'<div class="finding-evidence-title">Supporting Literature Grounding ({len(matched_papers)})</div>'
                        f'<div class="finding-evidence-list">{"".join(items_html)}</div>'
                        f'</div>'
                    )
                elif cited_tokens:
                    chips_html = "".join(f'<span class="inline-citation verdict-inline-citation">[{html.escape(t)}]</span>' for t in cited_tokens)
                    evidence_html = (
                        f'<div class="finding-evidence-box">'
                        f'<div class="finding-evidence-title">Cited References ({len(cited_tokens)})</div>'
                        f'<div>{chips_html}</div>'
                        f'</div>'
                    )

                st.markdown(
                    f"""
                    <div class="finding-card">
                        <div class="finding-card-header">
                            <span class="finding-num-badge">Finding {idx:02d}</span>
                        </div>
                        <div class="finding-statement">{statement_html}</div>
                        {evidence_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.write("")

        # Limitations & Open Research Gaps
        gaps = overview.verdict.limitations_and_gaps or []
        st.markdown(f"##### Limitations & Open Research Gaps ({len(gaps)})")
        if not gaps:
            st.markdown(
                '<div class="verdict-empty-notice">No critical methodological limitations or research gaps reported for this inquiry.</div>',
                unsafe_allow_html=True,
            )
        else:
            for gap in gaps:
                gap_html = format_finding_citations(gap, paper_lookup)
                st.markdown(
                    f"""
                    <div class="gap-card">
                        <div class="gap-tag">Open Challenge</div>
                        <div class="gap-content">{gap_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.write("")

        # Top Recommended Candidate Literature
        rec_pids = overview.verdict.top_recommended_papers or []
        if rec_pids:
            st.markdown(f"##### Top Recommended Literature ({len(rec_pids)})")
            st.caption("Highest-priority papers evaluated for methodology rigor, relevance, and evidence grounding.")
            for pid in rec_pids:
                p = paper_lookup.get(pid.lower())
                p_url = get_paper_external_url(pid, p)
                if p:
                    title_esc = html.escape(p.title)
                    if p_url:
                        title_html = f'<a href="{html.escape(p_url)}" target="_blank" rel="noopener noreferrer" class="verdict-rec-title">{title_esc}</a>'
                    else:
                        title_html = f'<span class="verdict-rec-title">{title_esc}</span>'

                    score_val = p.score if p.score is not None else getattr(p, "composite_score", None)
                    score_html = f'<span class="verdict-rec-score-badge">Score: {score_val:.2f}</span>' if score_val is not None else ""
                    authors_str = format_authors_short(p.authors)
                    venue_year = f"{html.escape(p.venue)} ({p.year})" if p.venue and p.year else (f"({p.year})" if p.year else (html.escape(p.venue) if p.venue else ""))
                    cites_html = f"<span>·</span><span>{p.citation_count:,} citations</span>" if p.citation_count is not None else ""

                    actions_html: list[str] = []
                    if p.doi:
                        doi_url = p.doi if p.doi.startswith("http") else f"https://doi.org/{p.doi.strip()}"
                        actions_html.append(f'<a href="{html.escape(doi_url)}" target="_blank" rel="noopener noreferrer" class="verdict-action-btn">DOI ↗</a>')
                    elif p_url and "doi.org" in p_url:
                        actions_html.append(f'<a href="{html.escape(p_url)}" target="_blank" rel="noopener noreferrer" class="verdict-action-btn">DOI ↗</a>')
                    if p_url and "arxiv.org" in p_url:
                        actions_html.append(f'<a href="{html.escape(p_url)}" target="_blank" rel="noopener noreferrer" class="verdict-action-btn">arXiv ↗</a>')
                    pdf_url = p.pdf_url or getattr(p, "open_access_pdf_url", None)
                    if pdf_url:
                        actions_html.append(f'<a href="{html.escape(pdf_url)}" target="_blank" rel="noopener noreferrer" class="verdict-action-btn">PDF ↗</a>')
                    actions_row = f'<div class="verdict-rec-actions">{"".join(actions_html)}</div>' if actions_html else ""

                    st.markdown(
                        f"""
                        <div class="verdict-rec-card">
                            <div class="verdict-rec-card-top">
                                <div class="verdict-rec-title-wrap">{title_html}</div>
                                <div class="verdict-rec-badges">
                                    <span class="verdict-rec-badge">★ Recommended</span>
                                    {score_html}
                                </div>
                            </div>
                            <div class="verdict-rec-meta">
                                <span>{authors_str}</span>
                                <span>·</span>
                                <span>{venue_year}</span>
                                {cites_html}
                            </div>
                            {actions_row}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    link_html = f'<a href="{html.escape(p_url)}" target="_blank" rel="noopener noreferrer" class="verdict-action-btn">View Record ↗</a>' if p_url else ""
                    st.markdown(
                        f"""
                        <div class="verdict-rec-card">
                            <div class="verdict-rec-card-top">
                                <code>{html.escape(pid)}</code>
                                <div class="verdict-rec-badges">
                                    <span class="verdict-rec-badge">★ Recommended Candidate</span>
                                </div>
                            </div>
                            {f'<div class="verdict-rec-actions">{link_html}</div>' if link_html else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.write("")

        # Expandable Narrative Synthesis
        if overview.synthesis:
            with st.expander("Evidence-Grounded Narrative Synthesis", expanded=False):
                st.markdown(
                    f'<div class="synthesis-box">{format_synthesis_citations(overview.synthesis, paper_lookup)}</div>',
                    unsafe_allow_html=True,
                )

        # Expandable Thematic Clusters
        if overview.themes:
            with st.expander(f"Thematic Literature Clusters ({len(overview.themes)})", expanded=False):
                theme_cols = st.columns(2) if len(overview.themes) > 1 else [st.container()]
                for idx, theme in enumerate(overview.themes):
                    col = theme_cols[idx % len(theme_cols)]
                    with col:
                        resolved_items: list[str] = []
                        for pid in theme.paper_ids[:3]:
                            p = paper_lookup.get(pid.lower())
                            if p:
                                venue_info = f" ({p.venue} {p.year})" if p.venue and p.year else (f" ({p.year})" if p.year else "")
                                title_snippet = p.title[:55] + ("..." if len(p.title) > 55 else "")
                                resolved_items.append(f"<li>{html.escape(title_snippet)}{venue_info}</li>")
                            else:
                                resolved_items.append(f"<li><code>{html.escape(pid)}</code></li>")

                        remaining = len(theme.paper_ids) - 3
                        if remaining > 0:
                            resolved_items.append(f"<li><em>+{remaining} additional papers</em></li>")

                        papers_html = f"<ul class='theme-paper-list'>{''.join(resolved_items)}</ul>" if resolved_items else ""

                        st.markdown(
                            f"""
                            <div class="theme-card">
                                <div class="theme-card-header">
                                    <span class="theme-card-num">Theme #{idx + 1}</span>
                                    <span class="theme-card-title">{html.escape(theme.name)}</span>
                                </div>
                                <div class="theme-card-summary">{html.escape(theme.summary)}</div>
                                <div class="theme-card-sub">Contributing papers ({len(theme.paper_ids)}):</div>
                                {papers_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
