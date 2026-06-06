import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import retrieve, chunks_parquet_exists, index_exists
from rag.generator import generate


def ask(
    query: str,
    top_k: int = 8,
    conferences: list[str] | None = None,
    years: list[int] | None = None,
) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, generate grounded answer.

    Args:
        query:       Natural language question.
        top_k:       Number of chunks to retrieve.
        conferences: Optional conference filter.
        years:       Optional year filter.

    Returns:
        Dict with keys:
            answer   — generated answer string
            sources  — list of source dicts (paper_title, conference, year, doi)
            chunks   — raw retrieved chunks with similarity scores
            grounded — True if answer is grounded in corpus chunks
    """
    chunks = retrieve(query, top_k=top_k, conferences=conferences, years=years)
    result = generate(query, chunks)
    result["chunks"] = chunks
    return result


def corpus_ready() -> bool:
    """Return True if the chunks parquet exists (index will be built on first query)."""
    return chunks_parquet_exists()
