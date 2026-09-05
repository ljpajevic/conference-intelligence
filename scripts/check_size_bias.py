"""
Corpus-size bias check for the paper relevance score.

The paper score is the mean of a venue's top-scoring paper similarities.
An order statistic, it matters from how many papers it is drawn from: if
the count of papers is a fixed number rather than a fixed fraction, venues
with more scrapped papers get a bonus unrelated to how well they fit the query.

This script separates fit from volume by capping every conference to the
same number of papers before scoring, then re-running the golden set.
Compare the capped and uncapped runs:

  - placements roughly unchanged  -> the ranking reflects fit
  - a venue collapses when capped -> it was being rewarded for volume

Findings so far:
  - With a fixed top-20, infocom (993 papers, 9.4x the smallest venue) took
    a top-3 slot in 9 of 10 golden cases; capped to 106 it fell to 2.3.
  - After switching to a fixed quantile (TOP_Q in agents/relevance_agent.py),
    the same comparison gives 2 uncapped against 2.7 capped.

Usage:
  python scripts/check_size_bias.py                 # cap at smallest venue
  python scripts/check_size_bias.py --cap 234       # cap at a chosen size
  python scripts/check_size_bias.py --seeds 5       # average over 5 subsamples
  python scripts/check_size_bias.py --alpha 1.0     # isolate the paper signal

Reads the golden set and the enriched parquet; writes nothing.
No LLM usage and no network, runs in seconds.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from agents.relevance_agent import ALPHA, _score_one_conference
from core.registry import list_conferences
from core.tools import compute_embeddings
from dashboard.persistence import load_data_state
from eval.metrics import mean, ndcg_at_k, precision_at_k
from eval.relevance_eval import load_golden

PARQUET = Path("data/enriched/networking_papers_enriched.parquet")


def rank_capped(df, cfp_data, user_embedding, description, alpha, cap, seed):
    """rank_conferences, but every conference capped to `cap` papers."""
    scores = []
    for conf in [c["name"] for c in list_conferences()]:
        conf_df = df[df["conference"].str.lower() == conf.lower()]
        if cap is not None and len(conf_df) > cap:
            conf_df = conf_df.sample(cap, random_state=seed)
        cfp_topics = cfp_data.get(conf, {}).get("topics", []) or []
        rec = _score_one_conference(
            conf, conf_df, cfp_topics,
            user_embedding, description, None,
            alpha=alpha, rationale=False,
        )
        scores.append((conf.lower(), rec["score"]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores]


def run(df, cfp_data, golden, embeddings, alpha, cap, seed, k):
    ndcgs, precs, topk = [], [], Counter()
    for case, user_embedding in zip(golden, embeddings):
        ranked = rank_capped(df, cfp_data, user_embedding,
                             case["research_description"], alpha, cap, seed)
        grades = {c.lower(): g for c, g in case["grades"].items()}
        relevant = {c for c, g in grades.items() if g >= 2}
        ndcgs.append(ndcg_at_k(ranked, grades, k))
        precs.append(precision_at_k(ranked, relevant, k))
        topk.update(ranked[:k])
    return mean(ndcgs), mean(precs), topk


def main():
    ap = argparse.ArgumentParser(
        description="Check whether the ranking rewards corpus size over fit."
    )
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help=f"paper weight (default: production ALPHA={ALPHA})")
    ap.add_argument("--cap", type=int, default=None,
                    help="papers per conference (default: smallest venue)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="subsamples to average over (default: 3)")
    ap.add_argument("-k", type=int, default=3, help="rank cutoff (default: 3)")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    state = load_data_state()
    cfp_data = state["result"].get("cfp_data", {}) if state else {}
    golden = load_golden()

    # embed each description once, not once per conference
    embeddings = compute_embeddings([c["research_description"] for c in golden])

    counts = df["conference"].str.lower().value_counts()
    cap = args.cap if args.cap is not None else int(counts.min())
    print(f"corpus:  {dict(counts)}")
    print(f"largest / smallest = {counts.max() / counts.min():.1f}x")
    print(f"alpha={args.alpha}  k={args.k}  cap={cap}  "
          f"cases={len(golden)}  seeds={args.seeds}\n")

    b_ndcg, b_prec, b_top = run(df, cfp_data, golden, embeddings,
                                args.alpha, None, 0, args.k)
    print(f"UNCAPPED  ndcg@{args.k}={b_ndcg:.4f}  precision@{args.k}={b_prec:.4f}")
    print(f"          top-{args.k}: {dict(b_top.most_common())}\n")

    ndcgs, precs, merged = [], [], Counter()
    for seed in range(args.seeds):
        n, p, t = run(df, cfp_data, golden, embeddings,
                      args.alpha, cap, seed, args.k)
        ndcgs.append(n)
        precs.append(p)
        merged.update(t)
        print(f"  seed {seed}: ndcg={n:.4f}  precision={p:.4f}")

    avg_top = {c: round(v / args.seeds, 1) for c, v in merged.most_common()}
    print(f"\nCAPPED    ndcg@{args.k}={mean(ndcgs):.4f}  "
          f"precision@{args.k}={mean(precs):.4f}")
    print(f"          top-{args.k} (mean over seeds): {avg_top}\n")

    print(f"{'venue':10}{'uncapped':>10}{'capped':>9}{'change':>9}")
    for conf in sorted(set(b_top) | set(merged),
                       key=lambda c: -b_top.get(c, 0)):
        u = b_top.get(conf, 0)
        c = merged.get(conf, 0) / args.seeds
        print(f"{conf:10}{u:10d}{c:9.1f}{c - u:+9.1f}")

    print("\nLarge negative change = that venue was rewarded for corpus size.")
    print("Values close to zero = placements reflect fit, not volume.")


if __name__ == "__main__":
    main()
