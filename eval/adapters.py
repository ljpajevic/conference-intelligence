import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """One retrieved chunk with its similarity score."""
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# RAG retrieval

def retrieve(question: str, top_k: int = 5) -> list[RetrievalResult]:
    """Query the ChromaDB collection the Insights tab uses."""
    from rag.retriever import retrieve as _retrieve
    chunks = _retrieve(question, top_k=top_k)
    return [
        RetrievalResult(
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=c["similarity"],
            metadata={
                "paper_title": c["paper_title"],
                "conference":  c["conference"],
                "year":        c["year"],
                "doi":         c["doi"],
            },
        )
        for c in chunks
    ]


# RAG answer generation

def generate_answer(question: str, chunks: list[RetrievalResult],
                    threshold: float | None = None) -> str | None:
    """Produce the grounded answer the Insights tab would show.

    Returns None when retrieval confidence is below threshold (suppressed).
    If threshold is None, uses the production default (MIN_SIMILARITY=0.25).
    """
    from rag.retriever import MIN_SIMILARITY
    from rag.generator import generate

    t = threshold if threshold is not None else MIN_SIMILARITY

    # apply threshold: filter out chunks below t
    filtered = [c for c in chunks if c.score >= t]

    # convert RetrievalResult back to the dict shape generator expects
    chunk_dicts = [
        {
            "text":        c.text,
            "paper_title": c.metadata.get("paper_title", ""),
            "conference":  c.metadata.get("conference", ""),
            "year":        c.metadata.get("year", ""),
            "doi":         c.metadata.get("doi", ""),
            "similarity":  c.score,
        }
        for c in filtered
    ]

    result = generate(question, chunk_dicts)
    return result["answer"] if result["grounded"] else None


# Relevance ranking

def rank_conferences(research_description: str,
                     paper_weight: float = 0.3,
                     cfp_weight: float = 0.7) -> list[str]:
    """Return conference names, best match first.
    Weights parameterised so the eval can sweep them.
    """
    from agents.relevance_agent import _score_one_conference, _build_local_llm
    from core.tools import compute_embeddings
    from core.registry import list_conferences
    import pandas as pd
    from pathlib import Path

    parquet_path = Path("data/enriched/networking_papers_enriched.parquet")
    df = pd.read_parquet(parquet_path)

    # load CFP data from the last data run pickle
    from dashboard.persistence import load_data_state
    state = load_data_state()
    cfp_data = state["result"].get("cfp_data", {}) if state else {}

    user_embedding = compute_embeddings([research_description])[0]
    llm = _build_local_llm()

    conferences = [c["name"] for c in list_conferences()]
    scores = []
    for conf_name in conferences:
        conf_df = df[df["conference"].str.lower() == conf_name.lower()]
        cfp_topics = cfp_data.get(conf_name, {}).get("topics", []) or []

        import agents.relevance_agent as ra
        orig_alpha = ra.ALPHA
        ra.ALPHA = paper_weight
        try:
            rec = _score_one_conference(
                conf_name, conf_df, cfp_topics,
                user_embedding, research_description, llm,
            )
        finally:
            ra.ALPHA = orig_alpha

        scores.append((conf_name, rec["score"]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores]

# shared: embedding function

_model = None

def embed(texts: list[str]):
    """all-MiniLM-L6-v2, same as the rest of the system."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(texts, normalize_embeddings=True)
