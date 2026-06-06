import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import chromadb
from chromadb.config import Settings

from core.tools import compute_embeddings

BASE_DIR    = Path(__file__).parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "chunks" / "networking_chunks.parquet"
CHROMA_DIR  = BASE_DIR / "data" / "chroma"
COLLECTION  = "conference_papers"

# similarity threshold below which retrieval is considered empty
MIN_SIMILARITY = 0.0 # 0.25

CHROMA_BATCH_SIZE = 5000

def _get_collection() -> chromadb.Collection:
    """Return the Chroma collection, creating and indexing it if needed."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        return client.get_collection(COLLECTION)

    print("[retriever] building Chroma index from chunks parquet...")
    df = _load_chunks()

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # embed in batches and add to collection
    texts = df["text"].tolist()
    embeddings = compute_embeddings(texts)

    # get embedgins in batches
    for start in range(0, len(df), CHROMA_BATCH_SIZE):
        batch = df.iloc[start:start + CHROMA_BATCH_SIZE]
        batch_embeddings = compute_embeddings(batch["text"].tolist())
        collection.add(
            ids=batch["chunk_id"].tolist(),
            embeddings=batch_embeddings.tolist(),
            documents=batch["text"].tolist(),
            metadatas=[
                {
                    "paper_title": row["paper_title"],
                    "conference":  row["conference"],
                    "year":        str(row["year"]),
                    "doi":         row["doi"] or "",
                }
                for _, row in batch.iterrows()
            ],
        )
        print(f"[retriever] indexed {min(start + CHROMA_BATCH_SIZE, len(df))}/{len(df)} chunks")
    return collection


def _load_chunks() -> pd.DataFrame:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks parquet not found at {CHUNKS_PATH}. "
            "Run the data pipeline first."
        )
    return pd.read_parquet(CHUNKS_PATH)


def retrieve(
    query: str,
    top_k: int = 8,
    conferences: list[str] | None = None,
    years: list[int] | None = None,
) -> list[dict]:
    """
    Embed query and retrieve top-k chunks from Chroma.

    Args:
        query:       Natural language question.
        top_k:       Number of chunks to retrieve.
        conferences: Optional list of conference names to filter by.
        years:       Optional list of years to filter by.

    Returns:
        List of dicts with keys: text, paper_title, conference, year, doi, similarity.
        Empty list if no results exceed MIN_SIMILARITY.
    """
    collection = _get_collection()

    query_embedding = compute_embeddings([query])[0].tolist()

    where: dict | None = None
    if conferences and years:
        where = {"$and": [
            {"conference": {"$in": [c.lower() for c in conferences]}},
            {"year": {"$in": [str(y) for y in years]}},
        ]}
    elif conferences:
        where = {"conference": {"$in": [c.lower() for c in conferences]}}
    elif years:
        where = {"year": {"$in": [str(y) for y in years]}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Chroma cosine distance = 1 - similarity
        similarity = 1.0 - dist
        if similarity < MIN_SIMILARITY:
            continue
        chunks.append({
            "text":        doc,
            "paper_title": meta["paper_title"],
            "conference":  meta["conference"],
            "year":        meta["year"],
            "doi":         meta["doi"],
            "similarity":  round(similarity, 3),
        })

    return chunks


def index_exists() -> bool:
    """Return True if the Chroma index has been built."""
    if not CHROMA_DIR.exists():
        return False
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return COLLECTION in [c.name for c in client.list_collections()]


def chunks_parquet_exists() -> bool:
    return CHUNKS_PATH.exists()
