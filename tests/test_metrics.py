"""Tests for eval/metrics.py.

Pure functions only: no models, no network, no filesystem. Every expected
value below is hand-computed from the metric definition, not from running
the implementation.
"""
import math

import pytest

from eval.metrics import hit_rate, mean, mrr, ndcg_at_k, precision_at_k


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------
def test_hit_rate_first_position():
    assert hit_rate(["a", "b", "c"], {"a"}) == 1.0


def test_hit_rate_last_position():
    assert hit_rate(["a", "b", "c"], {"c"}) == 1.0


def test_hit_rate_no_overlap():
    assert hit_rate(["a", "b", "c"], {"z"}) == 0.0


def test_hit_rate_empty_retrieved():
    assert hit_rate([], {"a"}) == 0.0


def test_hit_rate_empty_relevant():
    assert hit_rate(["a", "b"], set()) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------
def test_mrr_rank_one():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_rank_two():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5


def test_mrr_rank_three():
    assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_mrr_uses_first_relevant_only():
    # b at rank 2 and c at rank 3 are both relevant; only b counts
    assert mrr(["a", "b", "c"], {"b", "c"}) == 0.5


def test_mrr_no_relevant_retrieved():
    assert mrr(["a", "b"], {"z"}) == 0.0


def test_mrr_empty_retrieved():
    assert mrr([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------
def test_precision_all_relevant():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_partial():
    # a and c relevant in top 3 -> 2/3
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)


def test_precision_respects_k_cutoff():
    # only a is in the top 2 -> 1/2, c is cut off
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_precision_divides_by_k_not_list_length():
    # ranked is shorter than k; denominator stays k
    assert precision_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(0.4)


def test_precision_none_relevant():
    assert precision_at_k(["a", "b", "c"], {"z"}, 3) == 0.0


def test_precision_k_zero():
    assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0


def test_precision_k_negative():
    assert precision_at_k(["a", "b"], {"a"}, -1) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------
def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, 3) == pytest.approx(1.0)


def test_ndcg_reversed_ranking():
    # DCG  = 1/log2(2) + 2/log2(3) + 3/log2(4) = 1 + 1.261860 + 1.5 = 3.761860
    # IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.261860 + 0.5 = 4.761860
    dcg = 1 / math.log2(2) + 2 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(["c", "b", "a"], {"a": 3, "b": 2, "c": 1}, 3) == pytest.approx(dcg / idcg)
    # sanity: the reversed ranking scores clearly below perfect
    assert ndcg_at_k(["c", "b", "a"], {"a": 3, "b": 2, "c": 1}, 3) == pytest.approx(0.7900, abs=1e-3)


def test_ndcg_truncates_at_k():
    # ranked top-2 is [a, b]: a gains 1, b is ungraded (0) -> DCG = 1.0
    # ideal top-2 is [d, a]: 3/log2(2) + 1/log2(3) = 3 + 0.630930 = 3.630930
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["a", "b", "c", "d"], {"d": 3, "a": 1}, 2) == pytest.approx(1.0 / idcg)
    assert ndcg_at_k(["a", "b", "c", "d"], {"d": 3, "a": 1}, 2) == pytest.approx(0.2754, abs=1e-3)


def test_ndcg_all_zero_gains_returns_zero():
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 2) == 0.0


def test_ndcg_empty_gains_returns_zero():
    assert ndcg_at_k(["a", "b"], {}, 2) == 0.0


def test_ndcg_ungraded_items_score_zero():
    # nothing retrieved is in gains -> DCG 0, but IDCG > 0
    assert ndcg_at_k(["x", "y"], {"a": 3}, 2) == 0.0


def test_ndcg_graded_relevance_beats_binary_ordering():
    # putting the grade-3 item first must score higher than grade-2 first
    gains = {"a": 3, "b": 2, "c": 0}
    better = ndcg_at_k(["a", "b", "c"], gains, 3)
    worse = ndcg_at_k(["b", "a", "c"], gains, 3)
    assert better > worse


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------
def test_mean_basic():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_single_value():
    assert mean([0.5]) == 0.5


def test_mean_empty_returns_zero():
    assert mean([]) == 0.0


def test_mean_zeros():
    assert mean([0.0, 0.0]) == 0.0
