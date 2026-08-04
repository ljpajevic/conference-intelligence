import pandas as pd
from pathlib import Path

path = Path("data/enriched/networking_papers_enriched.parquet")
df = pd.read_parquet(path)
before = len(df)
df = df[df["title"].str.len() >= 5]
print(f"Dropped {before - len(df)} short-title rows")
df.to_parquet(path, index=False)
