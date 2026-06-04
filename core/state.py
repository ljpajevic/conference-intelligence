from typing import TypedDict, Annotated
import operator


# per-entry shapes

class DeadlineEntry(TypedDict):
    cycle: int          # for multi-cycle conferences e.g. 1, 2, ...
    label: str          # e.g. "Full Paper Submission", "Round 2"
    date: str           # ISO format YYYY-MM-DD


class CFPEntry(TypedDict):
    deadlines: list[DeadlineEntry]
    topics: list[str]
    url: str | None     # URL actually used to extract the data


class ClusterEntry(TypedDict):
    id: int
    label: str
    summary: str
    size: int
    representative_titles: list[str]


class TrajectoryEntry(TypedDict):
    interpretation: str
    counts: dict[str, dict[str, int]]   # {cluster_label: {year: paper_count}}


class TrendEntry(TypedDict):
    clusters: list[ClusterEntry]
    summaries: str          # conference-level summary
    trajectory: TrajectoryEntry

class RecommendationEntry(TypedDict):
    conference: str
    score: float             # combined score, primary ranking key
    paper_score: float       # topk paper similarity, 0-10
    cfp_score: float         # max CFP topic similarity, 0-10
    mean_score: float        # mean paper similarity, 0-10 (debugging signal)
    cfp_available: bool      # False if CFP scrape failed → score is paper-only
    rationale: str
    top_titles: list[str]
    top_cfp_topics: list[str]


# shared pipeline state

class PipelineState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph DAG.
    Each agent reads what it needs and writes its results back.
    """
    # ── Input ────────────────────────────────────────────────
    user_research_description: str        # user's research summary
    conferences_in_scope: list[str]       # conference names e.g. ["sigcomm", "imc"]
    years_in_scope: list[int]             # historical data years e.g. [2023, 2024, 2025]

    # ── Registry agent output ────────────────────────────────
    conference_metadata: dict             # name as a full conference record from DB

    # ── Paper Scraper agent output ───────────────────────────
    papers_df_path: str                   # path to enriched parquet file

    # ── CFP Scraper agent output ─────────────────────────────
    cfp_data: dict[str, CFPEntry]         # name to CFPEntry

    # ── Trend Analysis agent output ──────────────────────────
    trends: dict[str, TrendEntry]         # name to TrendEntry

    # ── Relevance agent output ───────────────────────────────
    recommendations: list[RecommendationEntry]

    # ── Errors & logs ────────────────────────────────────────
    errors: Annotated[list[str], operator.add]  # accumulates across all agents
