# AI Research Agent

**AI Research Agent** is a college mini-project that turns a research question into a
concise literature review. It helps users discover relevant research papers, understand
their main findings, compare them side by side, and decide which papers are worth
exploring further.

The application is built with **Streamlit** (Python) and is currently running on a
**mock data** backend, with a fully designed user interface and a complete automated
test suite. Integration with live paper-retrieval APIs and real AI analysis is being
handled by the rest of the team in parallel.

> **Current build notice**
> Everything visible in the interface today comes from the sample records in
> `mock_data.py`. No live API calls are made, no real AI model is invoked, and every
> score, overview, and verdict is a demonstration value. The interface is labelled
> accordingly (for example, *"Paper overview (demonstration)"*) so nothing on screen can
> be mistaken for a genuine research result.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [How to Run the Application](#how-to-run-the-application)
- [How to Run the Tests](#how-to-run-the-tests)
- [Current Implementation Status](#current-implementation-status)
- [Team Roles and Responsibilities](#team-roles-and-responsibilities)
- [Future Improvements](#future-improvements)
- [Limitations](#limitations)
- [Conclusion](#conclusion)

---

## Problem Statement

Reading a new research field is slow. A student or researcher typically searches on
several different portals, opens many PDFs, and still ends up with a scattered set of
papers and no clear answer to questions such as:

- Which papers actually address my research question?
- What is the main contribution of each paper?
- How do the papers differ, and which ones are worth my time?

Existing literature-review tools are either too heavy for a small project or require
paid accounts and API keys. Students often need a lightweight, transparent, and
free-to-run tool that turns a single research question into an organised, comparable
set of papers.

The AI Research Agent addresses this problem by combining two simple ideas:

1. **Retrieve** the papers that match a research topic.
2. **Analyse** each paper to produce a comparable summary, a transparent score, and a
   recommendation.

This repository contains the interface and testing work of that project, built on
demonstration data so the product can be designed, reviewed, and validated before the
live data and AI modules are connected.

---

## Project Objectives

- Provide a single input — a research topic — that returns a ranked, structured set of
  relevant papers.
- Make every paper easy to compare through a consistent card layout and a transparent,
  weighted score breakdown.
- Summarise each paper's research problem, main contribution, and related topics in a
  uniform, readable format.
- Give a clear, human-readable recommendation ("Keep this paper" / "Consider another
  paper") with an explanation for the decision.
- Show the weight and contribution of every scoring component so a user can understand
  how a score was produced.
- Deliver a clean, professional interface that works well on a normal laptop screen.
- Cover the implemented workflows with automated tests so the project can be extended
  safely.

---

## Key Features

The following features are **implemented and working in the current build**:

### Research search

- A single **Research topic** text field with a helpful placeholder example.
- Input validation: blank or whitespace-only topics are rejected with a clear warning.
- A simulated retrieval step with a live progress bar, a spinner, and an
  information-to-success status message.
- **Recent searches** are remembered in the sidebar (most recent first, up to five).

### Paper cards

Each result is presented as a self-contained card containing:

- Rank, paper title, and a highlighted overall score badge.
- **Year** and **Venue** tags, with the author list shown separately underneath.
- Compact metric cards for **Citations**, **Influential citations**, and the
  **Overall score (0-1)**, plus a progress bar for the score.
- A short summary of the paper.
- Links to the paper and to the PDF, when available.

### Score breakdown

- A transparent table of the five scoring components, each with its **score (0-1)**,
  its **weight**, and its **weighted contribution**.
- A **Weighted total** row, and a note explaining that missing components have their
  weight redistributed proportionally across the remaining components.

### Paper overview

- The paper's **Research problem**, **Main contribution**, and **Related topics**, with
  a caption stating that topic-specific relevance is not analysed in this build.

### Final verdict

- A recommendation ("Keep this paper" or "Consider another paper") with a short
  explanation of the reasoning behind it.

### Interface and navigation

- A polished **white-and-lavender** workspace theme with purple accents, defined in
  `ui/styles.css` and injected into the page at runtime.
- A sidebar with a brand header, navigation entries for the different result views,
  the recent-search list, and a reminder that the build uses demonstration data only.
- A discreet branded footer.
- A wide, responsive page layout (`layout="wide"`) so cards and metric tables have
  room to breathe.

---

## Technologies Used

| Technology | Version used | Purpose |
| --- | --- | --- |
| Python | 3.11.9 | Core language for the whole project |
| Streamlit | 1.65.0 | Frontend framework for the web interface |
| pytest | 9.1.1 | Automated testing of the implemented workflows |
| HTML / CSS | CSS 3 (vanilla) | Custom theme and card styling (`ui/styles.css`) |

No other third-party libraries, API clients, or AI SDKs are used in the current
build, and the project has no database or backend service.

---

## Project Structure

The project is intentionally small and flat so that it is easy to review and extend.

```text
ai_paprion/
│
├── app.py                 # Streamlit application: UI, rendering, scoring helpers,
│                          #   search flow, and the "if __name__ == '__main__'" entry point
├── mock_data.py           # Demonstration dataset (3 sample papers) and the scoring
│                          #   component weights and labels
├── requirements.txt       # Python dependencies (streamlit, pytest)
├── README.md              # This documentation
├── .gitignore             # Files excluded from version control
│
├── tests/
│   ├── __init__.py        # Marks the tests directory as a package
│   └── test_app.py        # Pytest suite (37 tests) covering the implemented workflows
│
└── ui/
    └── styles.css         # White-and-lavender theme: canvas, sidebar, cards, metric
                           #   cards, badges, tables, buttons, progress bars, footer
```

**Key files in detail**

- **`app.py`** — the single Streamlit entry point. It contains the page configuration,
  theme loading, sidebar rendering, the search flow, the paper-card renderer, the
  score-breakdown / overview / verdict expanders, and the small pure helper functions
  (formatting, score validation, weighted score computation) that the tests exercise
  directly.
- **`mock_data.py`** — holds `MOCK_RESULTS` (three sample papers with authors, venue,
  citation counts, scores, component scores, summaries, overviews, and verdicts) plus
  `SCORE_COMPONENT_WEIGHTS` and `SCORE_COMPONENT_LABELS`. The five weighted components
  are citation impact (0.40), influential citation ratio (0.20), venue quality (0.20),
  recency (0.10), and review score (0.10), which sum to 1.00.
- **`tests/test_app.py`** — the automated test suite. It tests the helper functions
  directly and uses Streamlit's `AppTest` utility to drive the real interface, for
  example clicking the **Search** button and asserting on the rendered expanders,
  captions, and result cards.
- **`ui/styles.css`** — a single stylesheet, loaded by `app.py` at runtime and
  injected with `st.markdown(..., unsafe_allow_html=True)`.

---

## Prerequisites

Before you begin, make sure you have:

- **Windows** with **PowerShell** (the commands below are PowerShell-ready).
- **Python 3.10 or newer** installed and available on your `PATH`
  (Python 3.11.9 was used during development).
  - Check with: `python --version`
- **Git**, if you want to clone the repository rather than copy the folder.
- An internet connection for the first-time installation of dependencies.

No database, API key, or external service account is required for the current build.

---

## Installation and Setup

Open **PowerShell** in the project folder and run the following commands in order.

### 1. Navigate to the project folder

```powershell
cd C:\Users\siddh_um\ai_paprion
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

> If PowerShell reports that script execution is disabled, allow local scripts once
> with:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Then run the activation command again. A `(venv)` prefix in the prompt confirms the
> environment is active.

### 4. Install the dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. (Optional) Verify the installation

```powershell
python -m pip list
```

You should see `streamlit` and `pytest` listed.

---

## How to Run the Application

With the virtual environment activated, start the Streamlit server from the project
folder:

```powershell
python -m streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. If it does not open
automatically, copy the printed local URL into your browser.

**Using the interface**

1. Type a research topic into the **Research topic** field, for example
   `retrieval-augmented generation`.
2. Press **Search**.
3. Watch the progress indicator while the (mock) retrieval step runs.
4. Scroll through the resulting paper cards, and expand **Score breakdown (…)**,
   **Paper overview (demonstration)**, and **Final verdict (demonstration)** for
   details.
5. Your topic is added to **Recent searches** in the sidebar.

**Useful Streamlit options**

```powershell
# Reload the browser tab automatically when the code changes
python -m streamlit run app.py --server.fileWatcherType "auto"

# Open on a specific port
python -m streamlit run app.py --server.port 8502
```

Press `Ctrl+C` in the PowerShell window to stop the server.

---

## How to Run the Tests

The test suite is written with **pytest** and uses Streamlit's `AppTest` utility to
exercise the real interface alongside direct unit tests of the helper functions.

```powershell
python -m pytest -q
```

**Expected output**

```text
.....................................                                    [100%]
37 passed
```

(The exact runtime varies between runs; the expected result is **37 passed** and no
failures.)

Other useful variations:

```powershell
# Show each test name
python -m pytest -v

# Run a single test
python -m pytest tests/test_app.py::test_search_still_renders_three_ranked_cards

# Run with coverage (install with: python -m pip install pytest-cov)
python -m pytest --cov=app --cov=mock_data
```

**What the tests cover**

- Mock dataset integrity (required fields, ranks, scores within the 0-1 range).
- Topic validation for blank and whitespace-only input.
- Search flow behaviour: warning on an empty topic, success message on a valid search,
  and three ranked paper cards rendered.
- Formatting helpers for counts, scores, and authors, including invalid and missing
  values.
- Score helpers: component validation, weighted contributions, weighted totals, and
  proportional weight redistribution when components are missing.
- Rendered expanders, score breakdown tables, overview sections, topic notes, and
  verdict text, including the demonstration disclaimers.

---

## Current Implementation Status

| Area | Status | Notes |
| --- | --- | --- |
| Streamlit interface | **Complete** | Wide layout, custom theme, sidebar, paper cards, metric cards, expanders, footer |
| Mock data layer | **Complete** | Three fully populated demonstration papers and the five scoring components |
| Scoring logic | **Complete** | Component validation, weighted contributions, weighted total, weight redistribution |
| Search flow (mock) | **Complete** | Validation, progress bar, spinner, status messages, recent searches |
| Automated tests | **Complete** | 37 pytest tests covering the implemented workflows |
| Live paper retrieval (APIs) | **Not integrated** | Owned by Member A; the application currently returns `mock_data.py` records only |
| Real AI analysis | **Not integrated** | Owned by Member B; overviews, summaries, and verdicts are demonstration text, not model output |
| End-to-end integration | **Pending** | Owned by Member C in collaboration with Members A and B |

**Branch status:** the current development branch for this work is `UI-&-testing`.
The work on this branch covers the interface and the test suite.

**In short:** the interface, the scoring model, and the tests are finished and
verified, while the live data and AI modules are still being built by the other team
members. The application is not yet connected to real scholarly sources, and live API
retrieval and real AI-generated analysis are **not confirmed as integrated**.

---

## Team Roles and Responsibilities

### Member A — Data & APIs

- Building the paper retrieval layer against sources such as **Semantic Scholar**,
  **OpenAlex**, and **arXiv**.
- Normalising the records returned by each source into a single, consistent paper
  format.
- Implementing **deduplication** so the same paper is not returned twice.
- Designing and implementing **caching** to reduce repeated calls to external APIs and
  keep the search fast.
- Exposing the retrieval layer to the rest of the team through a clear interface.

### Member B — AI & Logic

- Generating the **search query** from the user's research topic.
- Defining and implementing the **paper ranking** and scoring logic.
- Producing the **research analysis**: the research problem, the main contribution,
  and the related topics for each paper.
- Writing the **summaries** and the **recommendations** (the final verdicts) that the
  interface displays.
- Turning the raw analysis into the structured fields that the paper cards render.

### Member C — UI & Testing

- Designing and implementing the **Streamlit interface**: page layout, theme, sidebar,
  and navigation.
- Building the **paper cards**, **metric cards**, and the score breakdown, overview, and
  verdict **expanders**.
- Adding the **progress indicators** and status feedback for the search step.
- Writing and maintaining the **pytest** test suite for all implemented workflows.
- **Integrating** the modules delivered by Members A and B and keeping the interface
  correct as the backend changes.

---

## Future Improvements

- Connect the retrieval layer from Member A so results come from real scholarly
  sources instead of `mock_data.py`.
- Connect the AI analysis layer from Member B for genuine summaries, overviews, and
  verdicts.
- Let users filter results by year, venue, minimum citation count, and score.
- Add sorting and pagination for long result lists.
- Let users export a generated literature review as Markdown, PDF, or BibTeX.
- Support follow-up questions about a selected paper instead of a single query.
- Add a comparison view that places two or more papers side by side.
- Introduce user accounts and save search history and favourites between sessions.
- Add a downloadable BibTeX entry for every paper.
- Improve accessibility: stronger keyboard focus states and better colour contrast.
- Expand the test suite to cover the live retrieval and AI layers as they arrive, and
  add end-to-end tests for the integrated application.
- Add continuous integration so the tests run automatically on every push.

---

## Limitations

- **The data is mock data.** Results come from three hard-coded sample papers and are
  not related to the topic that was typed in.
- **No live API retrieval.** The application does not contact Semantic Scholar,
  OpenAlex, arXiv, or any other external service at this stage.
- **No real AI analysis.** Summaries, overviews, and verdicts are demonstration text
  written by hand; no language model is called.
- **No persistence.** Recent searches live in the browser session only and are lost on
  restart, and there is no database.
- **Search is simulated.** The progress bar and delay are a fixed placeholder for the
  real retrieval step, and the time it takes does not reflect real API latency.
- **Limited filtering.** There is no filtering, sorting, or pagination yet.
- **Scores are illustrative.** The weighted scoring model is fully implemented and
  tested, but the underlying component values are demonstration data.
- **Single-user and local.** The app runs on the local machine with no authentication
  or multi-user support.
- **Python 3.10+ required**, and the project currently targets desktop browsers.

---

## Conclusion

The AI Research Agent is a college mini-project that aims to make literature review
fast, structured, and transparent for students and early-stage researchers. The work
completed on the `UI-&-testing` branch delivers the full Streamlit interface — a
polished white-and-lavender workspace with a searchable paper-card layout, transparent
weighted score breakdowns, paper overviews, final verdicts, and clear demonstration
disclaimers — together with a 37-test pytest suite that verifies the workflows
implemented so far.

The interface is deliberately designed around the final product: it already accepts the
inputs, renders the outputs, and shows the progress feedback that the live backend will
drive. Once the retrieval and AI modules from Members A and B are connected and
verified, the same interface can be exercised against real data with no structural
changes. The project therefore stands as a working, testable, and presentable
foundation, with a clear roadmap for the remaining work.

---

*Maintained by the AI Research Agent team.*
*Repository: `ai_paprion` — branch `UI-&-testing`.*
