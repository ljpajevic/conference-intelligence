import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from core.state import PipelineState, RecommendationEntry
from core.tools import compute_embeddings


# config

TOP_K          = 20      # top-K most similar papers for paper score
TOP_TITLES     = 5       # titles included in the rationale prompt
TOP_CFP_TOPICS = 3       # CFP topics surfaced in rationale
SCORE_SCALE    = 10.0    # final score range [0, SCORE_SCALE]
ALPHA          = 0.3     # paper-weight: final = α*paper + (1-α)*cfp


# LLM setup

def _build_local_llm() -> ChatOllama:
    return ChatOllama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0,
        num_ctx=4096,
    )


# acoring

def _rescale(sim: float) -> float:
    """Linear rescale from cosine sim space [0, 0.7] to score space [0, SCORE_SCALE]."""
    return max(0.0, min(SCORE_SCALE, (sim / 0.7) * SCORE_SCALE))


def _compute_paper_scores(
    user_embedding: np.ndarray,
    conf_embeddings: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Returns (mean_score, topk_score, raw_sims)."""
    sims = conf_embeddings @ user_embedding   # both normalized -> cosine
    mean_sim = float(sims.mean())
    k = min(TOP_K, len(sims))
    topk_sim = float(np.sort(sims)[-k:].mean())
    return _rescale(mean_sim), _rescale(topk_sim), sims


def _compute_cfp_score(
    user_embedding: np.ndarray,
    topics: list[str],
) -> tuple[float, list[tuple[str, float]]]:
    """
    Returns (cfp_score, sorted_matches) where:
      cfp_score    — max user-vs-topic cosine sim, rescaled to 0-10
      matches      — list of (topic, sim) sorted desc by sim
    """
    if not topics:
        return 0.0, []

    topic_embeddings = compute_embeddings(topics)
    sims = topic_embeddings @ user_embedding      # shape (n_topics,)

    max_sim = float(sims.max())
    matches = sorted(
        zip(topics, sims.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return _rescale(max_sim), matches


# LLM rationale

_RATIONALE_PROMPT = """\
You are a research submission advisor.

A researcher's work is described as:
"{user_description}"

For the conference {conf_name}, here is the evidence:

PAPER EVIDENCE — the {n_titles} most topically similar past papers:
{titles}

CFP EVIDENCE — the {n_topics} most aligned topics from the upcoming CFP:
{cfp_topics}

Scores (0-10):
- Paper alignment: {paper_score:.1f}
- CFP alignment:   {cfp_score:.1f}
- Combined:        {final_score:.1f}

Write a 2-sentence rationale for whether the researcher should submit to \
{conf_name}. Ground every claim in the papers or topics above. Do NOT invent \
topics that are not represented in the evidence. If the score is low, say so honestly.

Return only the rationale text, no preamble.
"""


def _generate_rationale(
    llm: ChatOllama,
    user_description: str,
    conf_name: str,
    paper_score: float,
    cfp_score: float,
    final_score: float,
    top_titles: list[str],
    top_cfp_matches: list[tuple[str, float]],
) -> str:
    if not top_titles and not top_cfp_matches:
        return f"No evidence available for {conf_name}."

    titles_str = "\n".join(f"- {t}" for t in top_titles) or "(none)"
    cfp_str = (
        "\n".join(f"- {topic} (sim {sim:.2f})" for topic, sim in top_cfp_matches)
        or "(no CFP topics available)"
    )

    prompt = _RATIONALE_PROMPT.format(
        user_description=user_description,
        conf_name=conf_name.upper(),
        n_titles=len(top_titles),
        titles=titles_str,
        n_topics=len(top_cfp_matches),
        cfp_topics=cfp_str,
        paper_score=paper_score,
        cfp_score=cfp_score,
        final_score=final_score,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"  [relevance_agent] rationale error for {conf_name}: {e}")
        return f"(Rationale unavailable: {e})"


# per-conference scoring

def _score_one_conference(
    conf_name: str,
    conf_df: pd.DataFrame,
    cfp_topics: list[str],
    user_embedding: np.ndarray,
    user_description: str,
    llm: ChatOllama,
) -> RecommendationEntry:
    # paper signal
    if len(conf_df) > 0:
        embeddings = np.stack(conf_df["embedding"].to_numpy())
        mean_score, topk_score, sims = _compute_paper_scores(
            user_embedding, embeddings
        )
        top_idx = np.argsort(sims)[-TOP_TITLES:][::-1]
        top_titles = conf_df.iloc[top_idx]["title"].tolist()
    else:
        mean_score = topk_score = 0.0
        top_titles = []

    # CFP signal
    cfp_score, cfp_matches = _compute_cfp_score(user_embedding, cfp_topics)
    top_cfp_matches = cfp_matches[:TOP_CFP_TOPICS]

    # weighted
    # If CFP is missing, fall back to paper-only score
    if not cfp_topics:
        final_score = topk_score
        cfp_available = False
    else:
        final_score = ALPHA * topk_score + (1 - ALPHA) * cfp_score
        cfp_available = True

    # rationale
    rationale = _generate_rationale(
        llm, user_description, conf_name,
        topk_score, cfp_score, final_score,
        top_titles, top_cfp_matches,
    )

    print(
        f"  [relevance_agent] {conf_name}: "
        f"final={final_score:.2f} "
        f"(paper={topk_score:.2f}, cfp={cfp_score:.2f}"
        f"{', no CFP' if not cfp_available else ''})"
    )

    return {
        "conference":     conf_name,
        "score":          round(final_score, 2),
        "paper_score":    round(topk_score, 2),
        "cfp_score":      round(cfp_score, 2),
        "mean_score":     round(mean_score, 2),
        "cfp_available":  cfp_available,
        "rationale":      rationale,
        "top_titles":     top_titles,
        "top_cfp_topics": [t for t, _ in top_cfp_matches],
    }


# LangGraph node

def relevance_node(state: PipelineState) -> dict:
    """
    LangGraph node: rank conferences by combined paper + CFP relevance to
    the user's research description.

    final_score = α * paper_topk_score + (1 - α) * cfp_max_score    (default α = 0.3)

    Reads from state:
        user_research_description
        conferences_in_scope
        papers_df_path
        cfp_data

    Writes to state:
        recommendations — list sorted by final_score desc, each with:
            conference, score, paper_score, cfp_score, mean_score,
            cfp_available, rationale, top_titles, top_cfp_topics
        errors          — any per-conference errors
    """
    user_description: str = state["user_research_description"]
    conferences: list[str] = state["conferences_in_scope"]
    parquet_path: str = state["papers_df_path"]
    cfp_data: dict = state.get("cfp_data", {})

    print(f"\n[relevance_agent] ranking {len(conferences)} conferences (α={ALPHA})")
    print(f"  research: {user_description[:80]}…")

    if not parquet_path or not Path(parquet_path).exists():
        msg = "relevance_agent: papers_df_path missing or file not found"
        print(f"  ERROR {msg}")
        return {"recommendations": [], "errors": [msg]}

    df = pd.read_parquet(parquet_path)
    if "embedding" not in df.columns:
        msg = "relevance_agent: no 'embedding' column — run paper_pipeline first"
        print(f"  ERROR {msg}")
        return {"recommendations": [], "errors": [msg]}

    user_embedding = compute_embeddings([user_description])[0]

    llm = _build_local_llm()
    recommendations: list[RecommendationEntry] = []
    errors: list[str] = []

    for conf_name in conferences:
        conf_df = df[df["conference"].str.lower() == conf_name.lower()]
        cfp_topics = cfp_data.get(conf_name, {}).get("topics", []) or []

        try:
            rec = _score_one_conference(
                conf_name, conf_df, cfp_topics,
                user_embedding, user_description, llm,
            )
            recommendations.append(rec)
        except Exception as e:
            msg = f"relevance_agent [{conf_name}]: {e}"
            print(f"  ERROR {msg}")
            errors.append(msg)

    recommendations.sort(key=lambda r: r["score"], reverse=True)

    print("\n  [relevance_agent] rankings:")
    for r in recommendations:
        cfp_marker = "" if r["cfp_available"] else " (paper-only)"
        print(
            f"    {r['score']:>4.1f}  {r['conference']:<12}"
            f"  paper={r['paper_score']:.1f}  cfp={r['cfp_score']:.1f}{cfp_marker}"
        )
        print(f"          {r['rationale'][:90]}…")

    return {
        "recommendations": recommendations,
        "errors": errors,
    }
