import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import random
import time

import pandas as pd

from core.state import PipelineState

# pipeline internals that we reuse directly
from pipeline.paper_pipeline import (
    parse_dblp_xml,
    enrich_papers_async,
    create_chunks,
    save_dataframe,
    normalize_columns,
    print_coverage_report,
    add_embeddings,
    RAW_DIR,
    ENRICHED_DIR,
    CHUNKS_DIR,
)

# paths

ENRICHED_PATH = ENRICHED_DIR / "networking_papers_enriched.parquet"
RAW_PATH      = RAW_DIR      / "networking_papers_raw.parquet"
CHUNKS_PATH   = CHUNKS_DIR   / "networking_chunks.parquet"


# cache check

def _parquet_exists_and_covers(path: Path, conferences: list[str], years: list[int]) -> bool:
    """
    Cache hit if the parquet has AT LEAST ONE row for each requested conference.
    Years that turn up empty for some confs are treated as "no DBLP data" rather than as cache miss.
    """
    if not path.exists():
        return False

    try:
        df = pd.read_parquet(path, columns=["conference", "year"])
    except Exception:
        return False

    df["year"] = df["year"].astype(str)
    years_str = [str(y) for y in years]
    confs_norm = [c.lower() for c in conferences]

    # Ccche hit if every conference has at least 1 paper across the requested years
    for conf in confs_norm:
        conf_rows = df[(df["conference"] == conf) & (df["year"].isin(years_str))]
        if conf_rows.empty:
            print(f"  [paper_agent] cache miss: {conf} has no data for years {years}")
            return False

    return True


# scraping helpers

def _scrape_papers(conferences: list[str], years: list[int]) -> pd.DataFrame:
    from core.registry import get_conference
    from pipeline.paper_pipeline import parse_pacmnet_xml

    all_papers = []
    for conf_name in conferences:
        conf = get_conference(conf_name.lower())
        source = conf.get("dblp_source", "conf") if conf else "conf"
        conf_key = conf["dblp_key"] if conf else conf_name.lower()

        print(f"\n[paper_agent] scraping {conf_name} (source={source}) for years {years}")

        for year in sorted(years):
            if source == "pacmnet":
                papers = parse_pacmnet_xml(conf_name, year)
            else:
                papers = parse_dblp_xml(conf_name, conf_key, year)
            all_papers.extend(papers)
            time.sleep(random.uniform(3, 6))

        print(f"[paper_agent] cooling down after {conf_name}…")
        time.sleep(random.uniform(10, 20))

    return pd.DataFrame(all_papers)


# LangGraph node

def paper_scraper_node(state: PipelineState) -> dict:
    """
    LangGraph node: scrape papers and enrich them.

    Reads from state:
        conferences_in_scope  — list of conference short names
        years_in_scope        — list of integer years

    Writes to state:
        papers_df_path        — absolute path to the enriched parquet
        errors                — any error messages (appended, not replaced)
    """
    conferences: list[str] = state["conferences_in_scope"]
    years: list[int]       = state["years_in_scope"]

    print(f"\n[paper_agent] conferences={conferences}, years={years}")

    # 1. Cache check
    if _parquet_exists_and_covers(ENRICHED_PATH, conferences, years):
        print(f"[paper_agent] cache hit — using existing parquet: {ENRICHED_PATH}")

        # still chunk if chunks parquet is missing
        if not CHUNKS_PATH.exists():
            print(f"[paper_agent] chunks missing — building from cached parquet")
            try:
                df = pd.read_parquet(ENRICHED_PATH)
                chunkable = df[df["abstract"].notna()].copy()
                print(f"[paper_agent] chunking {len(chunkable)}/{len(df)} papers with abstracts")
                chunks_df = create_chunks(chunkable)
                CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
                save_dataframe(chunks_df, CHUNKS_PATH)
                print(f"[paper_agent] chunks saved: {len(chunks_df)} rows → {CHUNKS_PATH}")
            except Exception as exc:
                print(f"WARNING [paper_agent] chunking failed (non-fatal) — {exc}")

        return {"papers_df_path": str(ENRICHED_PATH)}

    print("[paper_agent] cache miss — starting full scrape + enrichment")

    # 2. Scrape raw papers from DBLp
    try:
        raw_df = _scrape_papers(conferences, years)
    except Exception as exc:
        msg = f"paper_agent: DBLP scrape failed — {exc}"
        print(f"ERROR {msg}")
        return {"papers_df_path": "", "errors": [msg]}

    if raw_df.empty:
        msg = "paper_agent: no papers collected — check conference keys and network"
        print(f"ERROR {msg}")
        return {"papers_df_path": "", "errors": [msg]}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_dataframe(raw_df, RAW_PATH)
    print(f"[paper_agent] raw papers saved: {len(raw_df)} rows → {RAW_PATH}")

    # 3. OpenAlex enrichment
    try:
        enriched_df = asyncio.run(enrich_papers_async(raw_df))
    except Exception as exc:
        msg = f"paper_agent: OpenAlex enrichment failed — {exc}"
        print(f"ERROR {msg}")
        return {"papers_df_path": "", "errors": [msg]}

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    save_dataframe(enriched_df, ENRICHED_PATH)
    print(f"[paper_agent] enriched papers saved: {len(enriched_df)} rows → {ENRICHED_PATH}")

    print_coverage_report(enriched_df)

    # 4. Embeddings
    enriched_df = add_embeddings(enriched_df)
    save_dataframe(enriched_df, ENRICHED_PATH)
    print(f"[paper_agent] embeddings added to parquet")

    # 5. Chunking
    try:
        chunkable = enriched_df[enriched_df["abstract"].notna()].copy()
        print(f"[paper_agent] chunking {len(chunkable)}/{len(enriched_df)} papers with abstracts")
        chunks_df = create_chunks(chunkable)
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        save_dataframe(chunks_df, CHUNKS_PATH)
        print(f"[paper_agent] chunks saved: {len(chunks_df)} rows → {CHUNKS_PATH}")
    except Exception as exc:
        # chunking failure is non-fatal; enriched parquet still usable
        msg = f"paper_agent: chunking failed (non-fatal) — {exc}"
        print(f"WARNING {msg}")
        return {"papers_df_path": str(ENRICHED_PATH), "errors": [msg]}

    return {"papers_df_path": str(ENRICHED_PATH)}
