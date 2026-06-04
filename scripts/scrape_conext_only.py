"""
Scrape that fetches only CoNEXT papers from PACMNET and appends to the
parquet. Useful for refreshing one conference without re-scraping everything.

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
from datetime import date

from pipeline.paper_pipeline import (
    parse_pacmnet_xml, enrich_papers_async, save_dataframe,
    add_embeddings, ENRICHED_DIR, N_YEARS,
)

def main():
    this_year = date.today().year
    years = list(range(this_year - N_YEARS, this_year + 1))

    all_papers = []
    for year in years:
        all_papers.extend(parse_pacmnet_xml("conext", year))

    if not all_papers:
        print("No papers found.")
        return

    new_df = pd.DataFrame(all_papers)
    print(f"Scraped {len(new_df)} CoNEXT papers via PACMNET")

    new_df = asyncio.run(enrich_papers_async(new_df))
    new_df = add_embeddings(new_df)

    parquet_path = ENRICHED_DIR / "networking_papers_enriched.parquet"
    existing = pd.read_parquet(parquet_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    save_dataframe(combined, parquet_path)
    print(f"Parquet now has {len(combined)} rows")

if __name__ == "__main__":
    main()
