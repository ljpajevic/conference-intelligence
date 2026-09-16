"""Build the slim corpus the deployed service reads.

Relevance scoring needs `embedding`, `title` and `conference`. This writes a
parquet with those plus `year` and `doi` for display and citation, leaving
the enriched corpus untouched for local use.

    python scripts/build_query_corpus.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import DATA_DIR

SOURCE = DATA_DIR / "enriched" / "networking_papers_enriched.parquet"
TARGET = DATA_DIR / "enriched" / "query_corpus.parquet"

KEEP = ["conference", "title", "year", "doi", "embedding"]

# named explicitly so a new column in the source does not silently ship
DROP_DELIBERATELY = ["abstract", "concepts"]


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"No enriched corpus at {SOURCE}. Run the data pipeline first.")

    df = pd.read_parquet(SOURCE)

    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise SystemExit(f"Source corpus is missing required column(s): {missing}")

    unexpected = [c for c in df.columns if c not in KEEP + DROP_DELIBERATELY]
    if unexpected:
        print(f"Dropping columns not in KEEP: {unexpected}")

    slim = df[KEEP].copy()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(TARGET, index=False)

    src_mb = SOURCE.stat().st_size / 1e6
    out_mb = TARGET.stat().st_size / 1e6
    print(f"Wrote {TARGET.relative_to(Path(__file__).parent.parent)}")
    print(f"  rows:     {len(slim)}")
    print(f"  columns:  {', '.join(KEEP)}")
    print(f"  dropped:  {', '.join(c for c in DROP_DELIBERATELY if c in df.columns)}")
    print(f"  size:     {src_mb:.1f} MB -> {out_mb:.1f} MB")

    assert "abstract" not in slim.columns, "abstract column survived the slice"


if __name__ == "__main__":
    main()
