import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json

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

    # if no conf specified, use all in DB
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


# build DAG

def build_graph():
    graph = StateGraph(PipelineState)

    # define nodes
    graph.add_node("registry", registry_node)
    graph.add_node("paper_scraper", paper_scraper_node)
    graph.add_node("cfp_scraper", cfp_scraper_node)
    graph.add_node("trend_analysis", trend_analysis_node)
    graph.add_node("relevance", relevance_node)

    # edges
    graph.set_entry_point("registry")
    graph.add_edge("registry", "paper_scraper")
    graph.add_edge("paper_scraper", "cfp_scraper")
    graph.add_edge("cfp_scraper", "trend_analysis")
    graph.add_edge("trend_analysis", "relevance")
    graph.add_edge("relevance", END)

    return graph.compile()


# run pipeline

def run_pipeline(
    user_research_description: str,
    conferences: list[str] = None,
    years: list[int] = None,
):
    graph = build_graph()

    initial_state: PipelineState = {
        "user_research_description": user_research_description,
        "conferences_in_scope": conferences or [],
        "years_in_scope": years or [2023, 2024, 2025],
        "conference_metadata": {},
        "papers_df_path": "",
        "cfp_data": {},
        "trends": {},
        "recommendations": [],
        "errors": [],
    }

    print("\n" + "="*50)
    print("CONFERENCE INTELLIGENCE PIPELINE")
    print("="*50)
    print(f"Research: {user_research_description}")
    print(f"Conferences: {conferences or 'all'}")
    print(f"Years: {years or [2023, 2024, 2025]}")

    result = graph.invoke(initial_state)

    print("\n[Pipeline complete]")
    print(f"Conferences loaded: {list(result['conference_metadata'].keys())}")
    print(f"Errors: {result['errors']}")

    return result


if __name__ == "__main__":
    result = run_pipeline(
        user_research_description="computational edge offloading for AR applications",
        conferences=["sigcomm", "mobicom", "mobisys", "imc"],
        years=[2023, 2024, 2025],
    )
