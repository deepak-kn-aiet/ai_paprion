import time
from pathlib import Path

import streamlit as st

from mock_data import (
    SCORE_COMPONENT_LABELS,
    SCORE_COMPONENT_WEIGHTS,
    get_mock_results,
)

THEME_CSS_PATH = Path(__file__).resolve().parent / "ui" / "styles.css"

SIDEBAR_NAV_ITEMS = [
    ("🏠", "Research search"),
    ("📄", "Paper cards"),
    ("📊", "Score breakdowns"),
    ("✅", "Final verdicts"),
]

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


def render_sidebar():
    st.sidebar.title("🔎 AI Research Agent")
    st.sidebar.caption("Modern AI SaaS Workspace")
    st.sidebar.markdown("**Navigation**")
    for index, (icon, label) in enumerate(SIDEBAR_NAV_ITEMS):
        nav_class = "nav-item nav-item-active" if index == 0 else "nav-item"
        st.sidebar.markdown(
            f'<div class="{nav_class}">{icon} {label}</div>',
            unsafe_allow_html=True,
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
        if is_valid_component_score(raw_score) and raw_score is not None:
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


def render_paper_card(card):
    with st.container(border=True):
        header_col, badge_col = st.columns([4, 1], vertical_alignment="center")
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
        render_metric_cards(card)
        if card["summary"]:
            st.write(card["summary"])
        render_score_breakdown(card)
        render_paper_overview(card)
        render_paper_verdict(card)
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


def render():
    st.set_page_config(
        page_title="AI Research Agent",
        page_icon=":mag:",
        layout="wide",
    )
    render_theme()
    render_sidebar()
    st.title("AI Research Agent")
    st.write(
        "AI Research Agent is a college mini-project that turns a research topic into a "
        "concise literature review. Enter a research question and the agent gathers "
        "relevant papers, summarizes their findings, and presents a structured review. "
        "This initial version uses mock data while the core modules are under development."
    )
    st.markdown('<div class="accent-divider"></div>', unsafe_allow_html=True)

    search_input_col, search_button_col = st.columns(
        [4, 1], vertical_alignment="bottom"
    )
    with search_input_col:
        topic = st.text_input(
            "Research topic",
            placeholder="e.g., retrieval-augmented generation",
        )
    with search_button_col:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked:
        if not is_valid_topic(topic):
            st.warning("Please enter a research topic before searching.")
        else:
            status_message = st.empty()
            status_message.info(f"Searching for papers about: {topic.strip()}")
            progress_bar = st.progress(0)
            with st.spinner("Searching scholarly sources (mock)..."):
                for step in range(1, 5):
                    time.sleep(0.25)
                    progress_bar.progress(step * 25)
            status_message.success("Mock search complete.")
            progress_bar.empty()
            record_recent_search(topic.strip())
            st.subheader("Mock Research Results")
            st.caption(DEMO_NOTE)
            for paper in mock_search(topic):
                render_paper_card(build_paper_card(paper, topic.strip()))

    render_footer()


if __name__ == "__main__":
    render()
