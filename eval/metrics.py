"""Retrieval and ranking metrics. Pure functions, no system dependencies."""

import math

def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1.0 if any relevant chunk appears in the retrieved list."""
    return 1.0 if any(r in relevant_ids for r in retrieved_ids) else 0.0

def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant chunk (0 if none retrieved)."""
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0

def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return sum(1 for x in top if x in relevant) / k


def ndcg_at_k(ranked: list[str], gains: dict[str, float], k: int) -> float:
    """NDCG@k with graded relevance.

    gains: item -> relevance grade (e.g. 3 = perfect fit, 1 = plausible,
    0/absent = irrelevant).
    """
    def dcg(items):
        return sum(gains.get(item, 0.0) / math.log2(i + 2)
                   for i, item in enumerate(items[:k]))

    ideal = sorted(gains, key=gains.get, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(ranked) / ideal_dcg


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
