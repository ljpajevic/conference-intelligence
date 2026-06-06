import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, END
from core.state import PipelineState
from core.registry import list_conferences, get_conference
from agents.paper_agent import paper_scraper_node
from agents.cfp_agent import cfp_scraper_node
from agents.trend_agent import trend_analysis_node
from agents.relevance_agent import relevance_node


def registry_node(state: PipelineState) -> PipelineState:
    """
    Loads conference metadata from the registry into shared state.
    Filters to only conferences in scope.
    """
    print("\n[Registry Agent] Loading conference metadata...")
    conferences_in_scope = state.get("conferences_in_scope", [])

    if not conferences_in_scope:
        all_confs = list_conferences()
        conferences_in_scope = [c["name"] for c in all_confs]
        print(f"  No scope specified — using all {len(conferences_in_scope)} conferences")

    metadata = {}
    errors = []
    for name in conferences_in_scope:
        conf = get_conference(name)
        if conf:
            metadata[name] = conf
            print(f"  ✓ {conf['full_name']}")
        else:
            msg = f"Conference '{name}' not found in registry"
            errors.append(msg)
            print(f"  ✗ {msg}")

    return {
        **state,
        "conferences_in_scope": list(metadata.keys()),
        "conference_metadata": metadata,
        "errors": errors,
    }


def build_data_graph():
    """
    Slow graph — run rarely.
    registry → paper_scraper → cfp_scraper → trend_analysis
    Produces parquet on disk and cached CFP data.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("registry", registry_node)
    graph.add_node("paper_scraper", paper_scraper_node)
    graph.add_node("cfp_scraper", cfp_scraper_node)
    graph.add_node("trend_analysis", trend_analysis_node)
    graph.set_entry_point("registry")
    graph.add_edge("registry", "paper_scraper")
    graph.add_edge("paper_scraper", "cfp_scraper")
    graph.add_edge("cfp_scraper", "trend_analysis")
    graph.add_edge("trend_analysis", END)
    return graph.compile()


def build_query_graph():
    """
    Fast graph — run per research description.
    registry → relevance
    Reads parquet and CFP data from disk; no scraping.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("registry", registry_node)
    graph.add_node("relevance", relevance_node)
    graph.set_entry_point("registry")
    graph.add_edge("registry", "relevance")
    graph.add_edge("relevance", END)
    return graph.compile()


def run_data_pipeline(
    conferences: list[str] = None,
    years: list[int] = None,
) -> dict:
    """
    Run the slow data collection pipeline.
    Returns state containing papers_df_path, cfp_data, trends.
    """
    graph = build_data_graph()
    initial_state: PipelineState = {
        "user_research_description": "",
        "conferences_in_scope": conferences or [],
        "years_in_scope": years or [2023, 2024, 2025],
        "conference_metadata": {},
        "papers_df_path": "",
        "cfp_data": {},
        "trends": {},
        "recommendations": [],
        "errors": [],
    }

    print("\n" + "=" * 50)
    print("CONFERENCE INTELLIGENCE — DATA PIPELINE")
    print("=" * 50)
    print(f"Conferences: {conferences or 'all'}")
    print(f"Years: {years or [2023, 2024, 2025]}")

    result = graph.invoke(initial_state)

    print("\n[Data pipeline complete]")
    print(f"Conferences loaded: {list(result['conference_metadata'].keys())}")
    print(f"Errors: {result['errors']}")
    return result


def run_recommendations(
    user_research_description: str,
    conferences: list[str] = None,
    papers_df_path: str = "",
    cfp_data: dict = None,
) -> dict:
    """
    Run the fast relevance scoring pipeline.
    Requires papers_df_path and cfp_data from a prior data pipeline run.
    """
    graph = build_query_graph()
    initial_state: PipelineState = {
        "user_research_description": user_research_description,
        "conferences_in_scope": conferences or [],
        "years_in_scope": [],
        "conference_metadata": {},
        "papers_df_path": papers_df_path,
        "cfp_data": cfp_data or {},
        "trends": {},
        "recommendations": [],
        "errors": [],
    }

    print("\n" + "=" * 50)
    print("CONFERENCE INTELLIGENCE — RECOMMENDATIONS")
    print("=" * 50)
    print(f"Research: {user_research_description}")
    print(f"Conferences: {conferences or 'all'}")

    result = graph.invoke(initial_state)

    print("\n[Recommendations complete]")
    print(f"Errors: {result['errors']}")
    return result


if __name__ == "__main__":
    # example: run data pipeline first, then recommendations
    data_result = run_data_pipeline(
        conferences=["sigcomm", "mobicom", "mobisys", "imc"],
        years=[2023, 2024, 2025],
    )
    rec_result = run_recommendations(
        user_research_description="computational edge offloading for AR applications",
        conferences=["sigcomm", "mobicom", "mobisys", "imc"],
        papers_df_path=data_result["papers_df_path"],
        cfp_data=data_result["cfp_data"],
    )
