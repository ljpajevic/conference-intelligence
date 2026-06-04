"""
One-off: add embeddings to the existing enriched parquet without
re-running DBLP/OpenAlex.

Usage:
  python scripts/add_embeddings_to_existing_parquet.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from core.tools import compute_embeddings, build_paper_text
from config import DATA_DIR


PARQUET_PATH = DATA_DIR / "enriched" / "networking_papers_enriched.parquet"


def main():
    if not PARQUET_PATH.exists():
        print(f"ERROR: parquet not found at {PARQUET_PATH}")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} papers from {PARQUET_PATH}")

    if "embedding" in df.columns and df["embedding"].notna().all():
        print("Embeddings already present — nothing to do.")
        return

    texts = [
        build_paper_text(row.get("title"), row.get("abstract"))
        for _, row in df.iterrows()
    ]
    print(f"Computing embeddings for {len(texts)} papers…")
    embeddings = compute_embeddings(texts)
    df["embedding"] = list(embeddings)

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved {len(df)} rows with embeddings → {PARQUET_PATH}")


if __name__ == "__main__":
    main()
