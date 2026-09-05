"""Markdown eval report with config fingerprint."""
import hashlib
import json
import platform
from datetime import date
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"


def config_fingerprint(extra: dict | None = None) -> str:
    payload = {
        "embedding_model": "all-MiniLM-L6-v2",
        "python": platform.python_version(),
        **(extra or {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _table(rows: list[dict]) -> str:
    if not rows:
        return "_no data_\n"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in (r[c] for c in cols)
        ) + " |")
    return "\n".join(lines) + "\n"


def write_report(rag_summary: dict | None,
                 threshold_rows: list[dict] | None,
                 ranking_summary: dict | None,
                 weight_rows: list[dict] | None,
                 deepeval_summary: dict | None = None,
                 config_extra: dict | None = None) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"eval_report_{date.today().isoformat()}.md"
    fp = config_fingerprint(config_extra)

    parts = [f"# Evaluation Report — {date.today().isoformat()}",
             f"\nConfig fingerprint: `{fp}`\n"]

    if rag_summary:
        parts.append("## RAG pipeline\n")
        parts.append(_table([{k: v for k, v in rag_summary.items() if k != "cases"}]))
        parts.append(f"\nCases evaluated: {rag_summary['n_cases']}. "
                     "`false_answer_rate` = answers produced for questions the corpus "
                     "cannot answer (lower is better); `answer_coverage` = answerable "
                     "questions actually answered (higher is better).\n")

    if deepeval_summary:
        parts.append("## DeepEval metrics\n")
        parts.append(_table([{k: v for k, v in deepeval_summary.items() if k != "cases"}]))
        parts.append(
            "\n`faithfulness` = fraction of answer claims supported by retrieved context "
            "(catches contradictions); `answer_relevancy` = how on-topic the answer is; "
            "`contextual_recall` = fraction of reference answer covered by retrieved chunks "
            "(requires `reference_answer` in golden set).\n"
        )

    if threshold_rows:
        parts.append("## Similarity-threshold calibration\n")
        parts.append(_table(threshold_rows))
        parts.append("\nProduction threshold should sit where false answers are "
                     "suppressed at minimal coverage cost.\n")

    if ranking_summary:
        parts.append("## Conference relevance ranking\n")
        skip = {"cases", "paper_only"}
        parts.append(_table([{k: v for k, v in ranking_summary.items()
                              if k not in skip}]))
        paper_only = ranking_summary.get("paper_only") or []
        if paper_only:
            parts.append(
                "\n**Scores are not comparable across this ranking.** "
                f"No CFP topics for: {', '.join(paper_only)}. "
                "These were scored on papers alone, under a different formula "
                "from the rest, and the paper/CFP weight does not apply to them.\n"
            )
        else:
            parts.append("\nAll conferences scored with the mixed paper + CFP "
                         "formula.\n")

    if weight_rows:
        parts.append("## Paper/CFP weight sweep\n")
        parts.append(_table(weight_rows))
        alpha = (config_extra or {}).get("alpha")
        prod = (f"Production uses {alpha}/{round(1 - alpha, 2)}"
                if alpha is not None else "Production weights")
        parts.append(f"\n{prod} — the sweep either validates or "
                     "improves that choice.\n")

    out.write_text("\n".join(parts))
    return out
