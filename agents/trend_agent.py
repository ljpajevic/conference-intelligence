import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from core.tools import compute_embeddings, build_paper_text

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.state import PipelineState
from core.cache import hash_content, cache_get, cache_set

from json_repair import repair_json

# config

N_CLUSTERS       = 5    # per conference
TOP_TITLES       = 2    # representative titles per cluster
MIN_PAPERS       = 5    # skip conference if fewer papers


# LLM

def _build_llm() -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=4096
    )

def _build_local_llm() -> ChatOllama:
    return ChatOllama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0,
        num_ctx=8192,
    )

# data loading

def _load_conference_papers(parquet_path: str, conf_name: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df[df["conference"].str.lower() == conf_name.lower()].copy()
    df["year"] = df["year"].astype(str)
    df["abstract"] = df["abstract"].fillna("")
    # Use title + abstract as the text to embed
    df["text"] = df["title"].fillna("") + ". " + df["abstract"]
    return df


# embedding + clustering

def _embed_and_cluster(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # use stored embeddings if available otherwise compute
    if "embedding" in df.columns and df["embedding"].notna().all():
        embeddings = np.stack(df["embedding"].to_numpy())
    else:
        print("  [trend_agent] no stored embeddings — computing now")
        texts = [build_paper_text(t, a) for t, a in zip(df["title"], df["abstract"])]
        embeddings = compute_embeddings(texts)

    n = min(N_CLUSTERS, len(df))
    km = KMeans(n_clusters=n, random_state=42, n_init="auto")
    df["cluster_id"] = km.fit_predict(embeddings)
    return df


def _cluster_summary(df: pd.DataFrame) -> list[dict]:
    """
    Build a compact cluster summary for the LLM prompt:
    cluster_id, size, representative titles (closest to centroid by count).
    """
    clusters = []
    for cid, group in df.groupby("cluster_id"):
        titles = group["title"].dropna().tolist()[:TOP_TITLES]
        clusters.append({
            "id":    int(cid),
            "size":  len(group),
            "titles": titles,
        })
    # Sort by size descending
    clusters.sort(key=lambda x: x["size"], reverse=True)
    return clusters


# LLM: label + summarise clusters

_LABEL_PROMPT = """\
You are a networking research analyst. Below are clusters of papers from {conf} ({years}).
Each cluster contains representative paper titles.

For each cluster, provide:
- "id": the cluster id (integer)
- "label": a short topic label (3-6 words)
- "summary": one sentence describing the research theme

Also provide:
- "conference_summary": 2-3 sentences summarising the overall research themes at {conf} across these years.

Return ONLY a valid JSON object in this exact shape — no preamble, no markdown fences:
{{
  "clusters": [
    {{"id": 0, "label": "...", "summary": "..."}},
    ...
  ],
  "conference_summary": "..."
}}

Clusters:
{clusters_json}
"""

def _llm_label_clusters(
    llm: ChatGroq,
    conf: str,
    years: list[str],
    clusters: list[dict],
) -> tuple[list[dict], str]:
    """
    Returns (labelled_clusters, conference_summary).
    Falls back to generic labels on any failure.
    """
    clusters_for_prompt = [
        {**c, "titles": [t[:60] for t in c.get("titles", [])]}
        for c in clusters
    ]

    prompt = _LABEL_PROMPT.format(
        conf=conf.upper(),
        years=", ".join(sorted(years)),
        clusters_json=json.dumps(clusters_for_prompt),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)], max_tokens=2048)
        parsed = json.loads(repair_json(response.content))

        labelled    = parsed.get("clusters", [])
        conf_summary = parsed.get("conference_summary", "")
        return labelled, conf_summary

    except Exception as e:
        print(f"  [trend_agent] label LLM error for {conf}: {e}")
        fallback = [{"id": c["id"], "label": f"Topic {c['id']}", "summary": ""} for c in clusters]
        return fallback, ""


# LLM: trajectory

_TRAJECTORY_PROMPT = """\
You are a networking research analyst. Below is year-over-year paper count data \
for topic clusters at {conf}, plus the top titles per cluster per year.

Interpret the trajectory:
- Which topics are growing?
- Which are declining or fading?
- Are there any emerging themes that only appeared recently?

Return ONLY a valid JSON object — no preamble, no markdown fences:
{{
  "interpretation": "2-4 sentence narrative of trends over time"
}}

Data:
{data_json}
"""

def _build_trajectory_counts(df: pd.DataFrame, labelled_clusters: list[dict]) -> dict:
    """
    Returns {{label: {{year: count, ...}, ...}}}
    """
    label_map = {c["id"]: c.get("label", f"Topic {c['id']}") for c in labelled_clusters}
    df = df.copy()
    df["label"] = df["cluster_id"].map(label_map)

    counts = {}
    for label, group in df.groupby("label"):
        counts[label] = group.groupby("year").size().to_dict()

    return counts


def _build_trajectory_titles(df: pd.DataFrame, labelled_clusters: list[dict]) -> dict:
    """
    Returns {{label: {{year: [title, ...], ...}}}} — top 3 titles per cluster per year.
    """
    label_map = {c["id"]: c.get("label", f"Topic {c['id']}") for c in labelled_clusters}
    df = df.copy()
    df["label"] = df["cluster_id"].map(label_map)

    result = {}
    for label, group in df.groupby("label"):
        result[label] = {}
        for year, ygroup in group.groupby("year"):
            result[label][year] = ygroup["title"].dropna().tolist()[:3]
    return result


def _llm_trajectory(
    llm: ChatGroq,
    conf: str,
    counts: dict,
    titles: dict,
) -> str:
    data = {"counts": counts, "top_titles": titles}
    prompt = _TRAJECTORY_PROMPT.format(
        conf=conf.upper(),
        data_json=json.dumps(data, indent=2),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(repair_json(response.content))
        return parsed.get("interpretation", "")

    except Exception as e:
        print(f"  [trend_agent] trajectory LLM error for {conf}: {e}")
        return ""


# per-conference analysis

def _analyse_conference(
    conf_name: str,
    parquet_path: str,
    llm: ChatGroq,
) -> dict:
    df = _load_conference_papers(parquet_path, conf_name)

    if len(df) < MIN_PAPERS:
        print(f"  [trend_agent] {conf_name}: only {len(df)} papers, skipping")
        return {"clusters": [], "summaries": "", "trajectory": {"interpretation": "", "counts": {}}}

    years = sorted(df["year"].unique().tolist())
    print(f"  [trend_agent] {conf_name}: {len(df)} papers, years={years}")

    # 1. embed + cluster (local, fast)
    df = _embed_and_cluster(df)
    raw_clusters = _cluster_summary(df)
    print(f"  [trend_agent] {conf_name}: {len(raw_clusters)} clusters formed")

    # 2. cache lookup: keyed on conference + the paper IDs that went into clustering
    paper_ids = sorted(df["title"].fillna("").tolist())
    cache_key = f"{conf_name}_{hash_content(paper_ids)}"
    cached = cache_get("trend_labels", cache_key)

    if cached:
        print(f"  [trend_agent] {conf_name}: cache hit → skipping LLM label call")
        labelled_clusters = cached["labelled_clusters"]
        conf_summary     = cached["conference_summary"]
    else:
        labelled_clusters, conf_summary = _llm_label_clusters(
            llm, conf_name, years, raw_clusters
        )
        # Only cache if the LLM actually succeeded (fallback returns "Topic N" labels)
        if conf_summary:
            cache_set("trend_labels", cache_key, {
                "labelled_clusters":   labelled_clusters,
                "conference_summary":  conf_summary,
            })
            print(f"  [trend_agent] {conf_name}: cached label result")

    # merge size + representative titles back in from raw clusters
    size_map = {c["id"]: c["size"] for c in raw_clusters}
    for c in labelled_clusters:
        c["size"] = size_map.get(c["id"], 0)
        raw = next((r for r in raw_clusters if r["id"] == c["id"]), {})
        c["representative_titles"] = raw.get("titles", [])

    # 3. build count/title data for trajectory (local, deterministic)
    counts = _build_trajectory_counts(df, labelled_clusters)
    titles = _build_trajectory_titles(df, labelled_clusters)

    # 4. trajectory LLM (Ollama, no quota concern, no caching needed)
    local_llm = _build_local_llm()
    interpretation = _llm_trajectory(local_llm, conf_name, counts, titles)

    return {
        "clusters": labelled_clusters,
        "summaries": conf_summary,
        "trajectory": {
            "interpretation": interpretation,
            "counts": counts,
        },
    }


# LangGraph node

def trend_analysis_node(state: PipelineState) -> dict:
    """
    LangGraph node: cluster papers per conference, label topics,
    and interpret year-over-year trajectory.

    Reads from state:
        conferences_in_scope  — list of conference short names
        papers_df_path        — path to enriched parquet

    Writes to state:
        trends   — {conf_name: {clusters, summaries, trajectory}}
        errors   — any per-conference errors (appended)
    """
    conferences: list[str] = state["conferences_in_scope"]
    parquet_path: str      = state["papers_df_path"]

    print(f"\n[trend_agent] analysing {len(conferences)} conferences")

    if not parquet_path or not Path(parquet_path).exists():
        msg = "trend_agent: papers_df_path missing or file not found"
        print(f"  ERROR {msg}")
        return {"trends": {}, "errors": [msg]}


    llm    = _build_llm()
    trends = {}
    errors = []

    for conf_name in conferences:
        print(f"\n  [trend_agent] processing {conf_name}…")
        try:
            trends[conf_name] = _analyse_conference(conf_name, parquet_path, llm)
        except Exception as e:
            msg = f"trend_agent [{conf_name}]: {e}"
            print(f"  ERROR {msg}")
            errors.append(msg)
            trends[conf_name] = {
                "clusters": [], "summaries": "",
                "trajectory": {"interpretation": "", "counts": {}}
            }

    return {"trends": trends, "errors": errors}
