import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


# config for embeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, local
EMBEDDING_DIM = 384

# lazy-loaded singleton
# first caller pays the load cost (1 sec), rest reuse
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  [tools] loading embedding model {EMBEDDING_MODEL_NAME}…")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def compute_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Compute L2-normalized embeddings for a list of texts.
    Returns an (n, EMBEDDING_DIM) numpy array, float32.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    embeddings = normalize(embeddings)         # cosine-friendly
    return embeddings.astype(np.float32)       # halves parquet size vs float64


def build_paper_text(title: str | None, abstract: str | None) -> str:
    """Canonical text representation for embedding a paper."""
    title = title or ""
    abstract = abstract or ""
    return f"{title}. {abstract}".strip()
