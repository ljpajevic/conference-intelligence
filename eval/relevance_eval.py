"""

Relevance ranking evaluation against expert-graded golden judgments.

Golden set format (eval/golden/relevance_golden.jsonl), one object per line:
    {
      "id": "r001",
      "research_description": "...",
      "grades": {"mobicom": 3, "mobisys": 3, "infocom": 2, "sigcomm": 1,
                 "imc": 1, "conext": 2, "eurosys": 0, "icdcs": 2}
    }

Grades: 3 = clear best fit, 2 = solid fit, 1 = plausible, 0 = wrong venue.

"""

import json
from pathlib import Path

from . import adapters
from .metrics import ndcg_at_k, precision_at_k, mean

GOLDEN_PATH = Path(__file__).parent / "golden" / "relevance_golden.jsonl"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def evaluate_ranking(k: int = 3, paper_weight: float | None = None, cfp_weight: float | None = None) -> dict:
    from agents.relevance_agent import ALPHA
    paper_weight = paper_weight if paper_weight is not None else ALPHA
    cfp_weight = cfp_weight if cfp_weight is not None else (1 - ALPHA)
    golden = load_golden()
    ndcgs, precs, per_case = [], [], []

    for case in golden:
        ranked = adapters.rank_conferences(case["research_description"],
                                           paper_weight=paper_weight,
                                           cfp_weight=cfp_weight)
        grades = {c.lower(): g for c, g in case["grades"].items()}
        ranked_l = [c.lower() for c in ranked]
        relevant = {c for c, g in grades.items() if g >= 2}
        n = ndcg_at_k(ranked_l, grades, k)
        p = precision_at_k(ranked_l, relevant, k)
        ndcgs.append(n)
        precs.append(p)
        per_case.append({"case_id": case["id"], f"ndcg@{k}": n,
                         f"precision@{k}": p, "top_ranked": ranked_l[:k]})
    return {
        "paper_weight": paper_weight,
        "cfp_weight": cfp_weight,
        f"ndcg@{k}": mean(ndcgs),
        f"precision@{k}": mean(precs),
        "cases": per_case,
    }

def weight_sweep(steps: int = 5, k: int = 3) -> list[dict]:
    """Sweep the paper/CFP weight mix. Production uses 0.3/0.7 -- this
    either validates that choice or finds a better one, with evidence.
    """
    rows = []
    for i in range(steps):
        pw = round(i / (steps - 1), 2)
        r = evaluate_ranking(paper_weight=pw, cfp_weight=round(1 - pw, 2), k=k)
        rows.append({"paper_weight": pw, "cfp_weight": round(1 - pw, 2),
                     f"ndcg@{k}": r[f"ndcg@{k}"],
                     f"precision@{k}": r[f"precision@{k}"]})
    return rows
