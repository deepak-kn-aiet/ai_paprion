import time
from pathlib import Path

import streamlit as st

from mock_data import (
    MOCK_RESULTS,
    SCORE_COMPONENT_LABELS,
    SCORE_COMPONENT_WEIGHTS,
    get_mock_results,
)

THEME_CSS_PATH = Path(__file__).resolve().parent / "ui" / "styles.css"

VIEW_SEARCH = "search"
VIEW_CARDS = "cards"
VIEW_ANALYTICS = "analytics"
VIEW_VERDICTS = "verdicts"

SIDEBAR_NAV_ITEMS = [
    ("🏠", "Research search", VIEW_SEARCH),
    ("📄", "Paper cards", VIEW_CARDS),
    ("📊", "Score breakdowns", VIEW_ANALYTICS),
    ("✅", "Final verdicts", VIEW_VERDICTS),
]

VIEW_LABELS = {
    VIEW_SEARCH: "Research search",
    VIEW_CARDS: "Paper cards",
    VIEW_ANALYTICS: "Score breakdowns",
    VIEW_VERDICTS: "Final verdicts",
}

CARD_SECTION_METRICS = "metrics"
CARD_SECTION_SUMMARY = "summary"
CARD_SECTION_BREAKDOWN = "breakdown"
CARD_SECTION_OVERVIEW = "overview"
CARD_SECTION_VERDICT = "verdict"
CARD_SECTION_LINKS = "links"

ALL_CARD_SECTIONS = (
    CARD_SECTION_METRICS,
    CARD_SECTION_SUMMARY,
    CARD_SECTION_BREAKDOWN,
    CARD_SECTION_OVERVIEW,
    CARD_SECTION_VERDICT,
    CARD_SECTION_LINKS,
)

GALLERY_CARD_SECTIONS = (
    CARD_SECTION_METRICS,
    CARD_SECTION_SUMMARY,
    CARD_SECTION_LINKS,
)

DEMO_NOTE = (
    "Demonstration results only - these sample papers are not "
    "dynamically retrieved for the entered topic."
)

SCORE_BREAKDOWN_NOTE = (
    "Component scores are demonstration values, not real evaluations. "
    "Each weighted contribution equals the component score multiplied by "
    "its weight. When a component is unavailable, its weight is "
    "redistributed proportionally across the remaining components."
)

OVERVIEW_NOTE = (
    "Demonstration content only - this overview is not a generated "
    "summary and was not extracted from a PDF."
)

VERDICT_NOTE = (
    "Demonstration recommendation only - not a genuine AI "
    "judgment, and not produced by an LLM, research search, "
    "PDF analysis, or topic relevance evaluation."
)


def load_theme_css():
    try:
        return THEME_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def render_theme():
    css = load_theme_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_active_view():
    view = st.session_state.get("active_view", VIEW_SEARCH)
    st.session_state["active_view"] = view
    return view


def set_active_view(view):
    st.session_state["active_view"] = view


def get_search_results():
    return st.session_state.get("search_results") or []


def store_search_results(cards, topic):
    st.session_state["search_results"] = cards
    st.session_state["search_topic"] = topic


def render_sidebar():
    st.sidebar.title("🔎 AI Research Agent")
    st.sidebar.caption("Modern AI SaaS Workspace")
    st.sidebar.markdown("**Navigation**")
    active_view = get_active_view()
    for icon, label, view in SIDEBAR_NAV_ITEMS:
        is_active = view == active_view
        st.sidebar.button(
            f"{icon}  {label}",
            key=f"nav-{view}",
            type="primary" if is_active else "tertiary",
            on_click=set_active_view,
            args=(view,),
            use_container_width=True,
        )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Recent searches**")
    recent_searches = st.session_state.get("recent_searches", [])
    if recent_searches:
        for recent_topic in recent_searches:
            st.sidebar.markdown(
                f'<div class="recent-search">{recent_topic}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("No searches yet.")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Demonstration build - mock data only. "
        "No live retrieval or AI analysis yet."
    )


def record_recent_search(topic):
    recent_searches = list(st.session_state.get("recent_searches", []))
    if topic not in recent_searches:
        recent_searches.insert(0, topic)
    st.session_state["recent_searches"] = recent_searches[:5]


def is_valid_topic(topic):
    return bool(topic and topic.strip())


def mock_search(topic):
    return sorted(get_mock_results(), key=lambda paper: paper.get("rank", 0))


def format_authors(paper):
    authors = paper.get("authors") or []
    if not authors:
        return "Unknown authors"
    return ", ".join(authors)


def format_count(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return "N/A"
    return f"{value:,}"


def format_score(score):
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "N/A"
    return f"{score:.2f} / 1.0"


def to_progress_value(score):
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return max(0.0, min(1.0, float(score)))


def is_valid_component_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0


def build_score_breakdown(paper):
    components = paper.get("score_components") or {}
    rows = []
    for name, weight in SCORE_COMPONENT_WEIGHTS.items():
        raw_score = components.get(name)
        if is_valid_component_score(raw_score):
            score = float(raw_score)
            rows.append(
                {
                    "component": SCORE_COMPONENT_LABELS[name],
                    "score": score,
                    "weight": weight,
                    "contribution": weight * score,
                    "available": True,
                }
            )
        else:
            rows.append(
                {
                    "component": SCORE_COMPONENT_LABELS[name],
                    "score": None,
                    "weight": weight,
                    "contribution": None,
                    "available": False,
                }
            )
    return rows


def compute_weighted_score(breakdown_rows):
    available = [row for row in breakdown_rows if row["available"]]
    if not available:
        return None
    total_weight = sum(row["weight"] for row in available)
    if total_weight <= 0:
        return None
    return sum(row["contribution"] for row in available) / total_weight


def build_paper_overview(paper):
    overview = paper.get("overview") or {}
    related_topics = overview.get("related_topics") or []
    return {
        "research_problem": overview.get("research_problem"),
        "main_contribution": overview.get("main_contribution"),
        "related_topics": related_topics,
        "available": bool(
            overview.get("research_problem")
            or overview.get("main_contribution")
            or related_topics
        ),
    }


def build_paper_verdict(paper):
    verdict = paper.get("verdict") or {}
    return {
        "recommendation": verdict.get("recommendation"),
        "explanation": verdict.get("explanation"),
        "available": bool(verdict.get("recommendation")),
    }


def build_paper_card(paper, topic=""):
    score_breakdown = build_score_breakdown(paper)
    return {
        "rank": paper.get("rank", "-"),
        "title": paper.get("title") or "Untitled",
        "authors": format_authors(paper),
        "year": paper.get("year", "N/A"),
        "venue": paper.get("venue") or "N/A",
        "citations": format_count(paper.get("citations")),
        "influential_citations": format_count(paper.get("influential_citations")),
        "score": format_score(paper.get("score")),
        "score_value": to_progress_value(paper.get("score")),
        "paper_link": paper.get("url"),
        "pdf_link": paper.get("pdf_url"),
        "summary": paper.get("summary") or "",
        "score_breakdown": score_breakdown,
        "weighted_total": compute_weighted_score(score_breakdown),
        "topic": topic,
        "overview": build_paper_overview(paper),
        "verdict": build_paper_verdict(paper),
    }


def render_score_breakdown(card):
    with st.expander(f"Score breakdown ({card['score']})"):
        st.caption(SCORE_BREAKDOWN_NOTE)
        lines = [
            "| Component | Score (0-1) | Weight | Weighted contribution |",
            "| --- | --- | --- | --- |",
        ]
        for row in card["score_breakdown"]:
            if row["available"]:
                lines.append(
                    f"| {row['component']} | {row['score']:.2f} | "
                    f"{row['weight']:.2f} | {row['contribution']:.3f} |"
                )
            else:
                lines.append(
                    f"| {row['component']} | N/A | {row['weight']:.2f} | N/A |"
                )
        if card["weighted_total"] is None:
            total_text = "N/A"
        else:
            total_text = f"{card['weighted_total']:.3f}"
        lines.append(f"| **Weighted total** | | **1.00** | **{total_text}** |")
        st.markdown("\n".join(lines))


def render_paper_overview(card):
    overview = card["overview"]
    with st.expander("Paper overview (demonstration)"):
        st.caption(OVERVIEW_NOTE)
        if not overview["available"]:
            st.write("Overview not available for this demonstration record.")
            return
        if overview["research_problem"]:
            st.markdown("**Research problem**")
            st.write(overview["research_problem"])
        if overview["main_contribution"]:
            st.markdown("**Main contribution**")
            st.write(overview["main_contribution"])
        if overview["related_topics"]:
            st.markdown("**Related demonstration topics**")
            st.write(", ".join(overview["related_topics"]))
        if card["topic"]:
            st.caption(
                f"Topic-specific relevance to "
                f"\"{card['topic']}\" is not analyzed "
                "in this demonstration version."
            )


def render_paper_verdict(card):
    verdict = card["verdict"]
    with st.expander("Final verdict (demonstration)"):
        st.caption(VERDICT_NOTE)
        if not verdict["available"]:
            st.write("Verdict not available for this demonstration record.")
            return
        st.markdown(f"**Verdict:** {verdict['recommendation']}")
        if verdict["explanation"]:
            st.write(verdict["explanation"])


def render_metric_cards(card):
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Citations</span>'
            f'<span class="metric-value">{card["citations"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Influential citations</span>'
            f'<span class="metric-value">{card["influential_citations"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with metric_cols[2]:
        st.markdown(
            '<div class="metric-card">'
            '<span class="metric-label">Overall score (0-1)</span>'
            f'<span class="metric-value">{card["score"]}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        if card["score_value"] is not None:
            st.progress(card["score_value"])


def render_paper_card(card, sections=None):
    visible = set(ALL_CARD_SECTIONS if sections is None else sections)
    with st.container(border=True):
        header_col, badge_col = st.columns([4, 1], vertical_alignment="top")
        with header_col:
            st.markdown(f"**Rank #{card['rank']}: {card['title']}**")
            st.markdown(
                '<div class="tag-row">'
                f'<span class="tag tag-year">{card["year"]}</span>'
                f'<span class="tag tag-venue">{card["venue"]}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        with badge_col:
            st.markdown(
                '<div class="badge-wrap">'
                f'<span class="score-badge">{card["score"]}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        st.caption(card["authors"])
        if CARD_SECTION_METRICS in visible:
            render_metric_cards(card)
        if CARD_SECTION_SUMMARY in visible and card["summary"]:
            st.write(card["summary"])
        if CARD_SECTION_BREAKDOWN in visible:
            render_score_breakdown(card)
        if CARD_SECTION_OVERVIEW in visible:
            render_paper_overview(card)
        if CARD_SECTION_VERDICT in visible:
            render_paper_verdict(card)
        if CARD_SECTION_LINKS in visible:
            links = []
            if card["paper_link"]:
                links.append(f"[Paper link]({card['paper_link']})")
            if card["pdf_link"]:
                links.append(f"[PDF link]({card['pdf_link']})")
            if links:
                st.markdown("  \n".join(links))
            else:
                st.caption("No paper or PDF links available for this record.")


def render_footer():
    st.markdown(
        '<div class="app-footer">'
        '<span class="footer-brand">AI Research Agent</span>'
        '<span class="footer-sep">·</span>'
        "<span>Demonstration build - mock data only</span>"
        "</div>",
        unsafe_allow_html=True,
    )


HEADER_MARK_SVG = (
    '<svg class="app-header-icon" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true" focusable="false">'
    '<path d="M12 5.2a2.9 2.9 0 0 0-5.3 1.3 2.7 2.7 0 0 0-1.5 4.4'
    ' 2.8 2.8 0 0 0 .9 4.2A2.9 2.9 0 0 0 9 18.8a3 3 0 0 0 3 1.1'
    ' 3 3 0 0 0 3-1.1 2.9 2.9 0 0 0 2.9-3.7 2.8 2.8 0 0 0 .9-4.2'
    ' 2.7 2.7 0 0 0-1.5-4.4A2.9 2.9 0 0 0 12 5.2Z"/>'
    '<path d="M12 5.2v14.7"/>'
    '<path d="M8.6 9.4h2.1"/>'
    '<path d="M13.3 13.2h2.1"/>'
    '<circle cx="8.6" cy="9.4" r="1.15" fill="currentColor" stroke="none"/>'
    '<circle cx="15.4" cy="13.2" r="1.15" fill="currentColor" stroke="none"/>'
    "</svg>"
)


def render_header(view):
    st.markdown(
        '<div class="app-header">'
        '<div class="app-header-main">'
        f'<span class="app-header-mark">{HEADER_MARK_SVG}</span>'
        '<div class="app-header-text">'
        '<div class="app-header-title">AI Research Agent</div>'
        '<div class="app-header-subtitle">'
        "Turn a research question into a structured literature review"
        "</div>"
        "</div>"
        "</div>"
        '<div class="app-header-meta">'
        f'<span class="app-header-chip">{VIEW_LABELS[view]}</span>'
        '<span class="app-header-note">Demonstration build - mock data only</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_empty_state(title, message):
    st.markdown(
        '<div class="empty-state">'
        f'<div class="empty-state-title">{title}</div>'
        f'<div class="empty-state-text">{message}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def run_mock_search(topic):
    status_message = st.empty()
    status_message.info(f"Searching for papers about: {topic}")
    progress_bar = st.progress(0)
    with st.spinner("Searching scholarly sources (mock)..."):
        for step in range(1, 5):
            time.sleep(0.25)
            progress_bar.progress(step * 25)
    status_message.success("Mock search complete.")
    progress_bar.empty()
    record_recent_search(topic)
    cards = [build_paper_card(paper, topic) for paper in mock_search(topic)]
    store_search_results(cards, topic)


def render_analytics_summary(cards):
    totals = [
        card["weighted_total"]
        for card in cards
        if card["weighted_total"] is not None
    ]
    average = sum(totals) / len(totals) if totals else None
    average_text = f"{average:.2f}" if average is not None else "N/A"
    component_scores = {}
    for card in cards:
        for row in card["score_breakdown"]:
            if row["available"]:
                component_scores.setdefault(row["component"], []).append(
                    row["score"]
                )
    st.markdown(
        '<div class="stat-grid">'
        '<div class="stat-tile">'
        '<span class="stat-label">Papers scored</span>'
        f'<span class="stat-value">{len(cards)}</span>'
        "</div>"
        '<div class="stat-tile">'
        '<span class="stat-label">Average weighted total</span>'
        f'<span class="stat-value">{average_text}</span>'
        "</div>"
        '<div class="stat-tile">'
        '<span class="stat-label">Components tracked</span>'
        f'<span class="stat-value">{len(component_scores)}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    bars = []
    for name, values in component_scores.items():
        mean = sum(values) / len(values)
        width = max(0.0, min(1.0, mean)) * 100
        bars.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{name}</span>'
            '<span class="bar-track">'
            f'<span class="bar-fill" style="width:{width:.0f}%"></span>'
            "</span>"
            f'<span class="bar-value">{mean:.2f}</span>'
            "</div>"
        )
    if bars:
        st.markdown(
            '<div class="bar-list">' + "".join(bars) + "</div>",
            unsafe_allow_html=True,
        )


def render_verdict_summary(cards):
    counts = {}
    for card in cards:
        recommendation = card["verdict"]["recommendation"]
        if recommendation:
            counts[recommendation] = counts.get(recommendation, 0) + 1
    tiles = []
    for recommendation, count in counts.items():
        tiles.append(
            '<div class="stat-tile">'
            '<span class="stat-label">Recommendation</span>'
            f'<span class="stat-value stat-value-text">{recommendation}</span>'
            f'<span class="stat-meta">{count} paper(s) in this result set</span>'
            "</div>"
        )
    if tiles:
        st.markdown(
            '<div class="stat-grid">' + "".join(tiles) + "</div>",
            unsafe_allow_html=True,
        )


def render_search_view():
    st.markdown(
        '<div class="panel-heading">'
        '<div class="panel-heading-title">Start a new literature review</div>'
        '<div class="panel-heading-text">'
        "Enter a research topic to build the demonstration result set. "
        "The agent returns ranked sample papers with scores, overviews, "
        "and recommendations."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    search_input_col, search_button_col = st.columns(
        [4, 1], vertical_alignment="bottom"
    )
    with search_input_col:
        if "research_topic" not in st.session_state:
            st.session_state["research_topic"] = st.session_state.get(
                "search_topic", ""
            )
        topic = st.text_input(
            "Research topic",
            key="research_topic",
            placeholder="e.g., retrieval-augmented generation",
        )
    with search_button_col:
        search_clicked = st.button(
            "Search",
            use_container_width=True,
            type="primary",
        )

    if search_clicked:
        if not is_valid_topic(topic):
            st.warning("Please enter a research topic before searching.")
        else:
            run_mock_search(topic.strip())

    results = get_search_results()
    if not results:
        return
    st.caption(
        "Showing demonstration results for: "
        f"{st.session_state.get('search_topic', '')}"
    )
    st.subheader("Mock Research Results")
    st.caption(DEMO_NOTE)
    for card in results:
        render_paper_card(card)


def render_cards_view():
    results = get_search_results()
    if not results:
        render_empty_state(
            "No paper cards yet",
            "Run a search from the Research search page to build the "
            "demonstration paper cards, then come back here to browse them.",
        )
        return
    st.subheader("Paper cards")
    st.caption(DEMO_NOTE)
    for card in results:
        render_paper_card(card, sections=GALLERY_CARD_SECTIONS)


def render_analytics_view():
    results = get_search_results()
    if not results:
        render_empty_state(
            "No score breakdowns yet",
            "Run a search from the Research search page to generate the "
            "demonstration score breakdowns shown on this page.",
        )
        return
    st.subheader("Score breakdowns")
    st.caption(SCORE_BREAKDOWN_NOTE)
    render_analytics_summary(results)
    for card in results:
        render_score_breakdown(card)


def render_verdicts_view():
    results = get_search_results()
    if not results:
        render_empty_state(
            "No final verdicts yet",
            "Run a search from the Research search page to generate the "
            "demonstration recommendations shown on this page.",
        )
        return
    st.subheader("Final verdicts")
    st.caption(VERDICT_NOTE)
    render_verdict_summary(results)
    for card in results:
        render_paper_verdict(card)


def render():
    st.set_page_config(
        page_title="AI Research Agent",
        page_icon=":mag:",
        layout="wide",
    )
    render_theme()
    render_sidebar()
    view = get_active_view()
    render_header(view)
    if view == VIEW_SEARCH:
        render_search_view()
    elif view == VIEW_CARDS:
        render_cards_view()
    elif view == VIEW_ANALYTICS:
        render_analytics_view()
    elif view == VIEW_VERDICTS:
        render_verdicts_view()
    render_footer()


if __name__ == "__main__":
    render()
