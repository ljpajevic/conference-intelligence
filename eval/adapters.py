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


@dataclass
class RankedConference:
    """One scored conference, carrying how its score was computed.

    cfp_available is False when the conference had no CFP topics, in which
    case _score_one_conference falls back to a paper-only score — a
    different formula from the rest of the ranking.
    """
    conference: str
    score: float
    cfp_available: bool


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

def apply_threshold(chunks: list[RetrievalResult],
                    threshold: float | None = None) -> list[RetrievalResult]:
    """Chunks actually provided to the generator at this threshold.

    Grade answers against these chunks, not raw retrieval results.
    """
    from rag.retriever import MIN_SIMILARITY
    t = threshold if threshold is not None else MIN_SIMILARITY
    return [c for c in chunks if c.score >= t]


def generate_answer(question: str, chunks: list[RetrievalResult],
                    threshold: float | None = None) -> str | None:
    """Produce the grounded answer the Insights tab would show.

    Returns None when retrieval confidence is below threshold (suppressed).
    If threshold is None, uses the production default (MIN_SIMILARITY=0.60).
    """
    from rag.generator import generate

    filtered = apply_threshold(chunks, threshold)

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
                     paper_weight: float | None = None) -> list[RankedConference]:
    """Return conferences, best match first.

    paper_weight is the alpha the scorer applies; the CFP score is always (1 - alpha).
    Defaults to production ALPHA.
    """
    from agents.relevance_agent import ALPHA, _score_one_conference
    from core.tools import compute_embeddings
    from core.registry import list_conferences
    from config import DATA_DIR
    import pandas as pd

    if paper_weight is None:
        paper_weight = ALPHA

    parquet_path = DATA_DIR / "enriched" / "networking_papers_enriched.parquet"
    df = pd.read_parquet(parquet_path)

    # load CFP data from the last data run pickle
    from dashboard.persistence import load_data_state
    state = load_data_state()
    cfp_data = state["result"].get("cfp_data", {}) if state else {}

    user_embedding = compute_embeddings([research_description])[0]

    conferences = [c["name"] for c in list_conferences()]
    scores = []
    for conf_name in conferences:
        conf_df = df[df["conference"].str.lower() == conf_name.lower()]
        cfp_topics = cfp_data.get(conf_name, {}).get("topics", []) or []

        rec = _score_one_conference(
            conf_name, conf_df, cfp_topics,
            user_embedding, research_description, None,
            alpha=paper_weight,
            rationale=False,
        )

        scores.append(RankedConference(
            conference=conf_name,
            score=rec["score"],
            cfp_available=rec["cfp_available"],
        ))

    scores.sort(key=lambda r: r.score, reverse=True)
    return scores

# shared: embedding function

_model = None

def embed(texts: list[str]):
    """all-MiniLM-L6-v2, same as the rest of the system."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(texts, normalize_embeddings=True)
