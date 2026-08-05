"""RAG evaluation: retrieval quality, groundedness, threshold calibration.

Golden set format (eval/golden/rag_golden.jsonl), one JSON object per line:
    {
      "id": "q001",
      "question": "...",
      "relevant_chunk_ids": ["sigcomm-2024-...", ...],   # optional but enables hit rate / MRR
      "reference_answer": "...",                          # optional, for your own review
      "answerable": true                                  # false = corpus does NOT contain the answer
    }

Include BOTH answerable and unanswerable questions.
Unanswerable ones are what the similarity threshold exists for, a good threshold suppresses them.

"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from . import adapters
from .metrics import hit_rate, mrr, mean

GOLDEN_PATH = Path(__file__).parent / "golden" / "rag_golden.jsonl"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


# Groundedness: is each answer sentence supported by a retrieved chunk?


def groundedness_embedding(answer: str, chunks: list[adapters.RetrievalResult],
                           support_threshold: float = 0.55) -> float:
    """Fraction of answer sentences whose max cosine similarity against any
    retrieved chunk exceeds support_threshold. Cheap, deterministic, no LLM.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    if not sentences or not chunks:
        return 0.0
    sent_emb = adapters.embed(sentences)
    chunk_emb = adapters.embed([c.text for c in chunks])
    sims = sent_emb @ chunk_emb.T          # normalized embeddings -> cosine
    supported = (sims.max(axis=1) >= support_threshold).sum()
    return float(supported) / len(sentences)


def groundedness_llm_judge(question: str, answer: str,
                           chunks: list[adapters.RetrievalResult]) -> float | None:
    """Optional second opinion via local Ollama (quota-free). Returns a score
    in [0,1] or None if the judge is unavailable/unparseable.
    """
    try:
        from langchain_ollama import ChatOllama
        judge = ChatOllama(model="llama3.1:8b", temperature=0)
        context = "\n---\n".join(c.text for c in chunks)
        prompt = (
            "You are grading whether an answer is fully supported by the context.\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}\n\n"
            "Reply with ONLY a number between 0 and 1: the fraction of the answer's "
            "claims that the context supports. No other text."
        )
        raw = judge.invoke(prompt).content.strip()
        m = re.search(r"(?:0?\.\d+|[01](?:\.0+)?)", raw)
        return min(1.0, max(0.0, float(m.group()))) if m else None
    except Exception:
        return None


# core evaluation

@dataclass
class RagCaseResult:
    case_id: str
    answerable: bool
    answered: bool
    hit: float | None
    rr: float | None
    groundedness: float | None
    judge_score: float | None


def evaluate_rag(top_k: int = 5, threshold: float | None = None,
                 use_llm_judge: bool = False) -> dict:
    golden = load_golden()
    results: list[RagCaseResult] = []

    for case in golden:
        chunks = adapters.retrieve(case["question"], top_k=top_k)
        answer = adapters.generate_answer(case["question"], chunks, threshold=threshold)
        answered = answer is not None

        relevant = set(case.get("relevant_chunk_ids", []))
        hit = hit_rate([c.chunk_id for c in chunks], relevant) if relevant else None
        rr = mrr([c.chunk_id for c in chunks], relevant) if relevant else None

        g = groundedness_embedding(answer, chunks) if answered else None
        j = groundedness_llm_judge(case["question"], answer, chunks) \
            if (answered and use_llm_judge) else None

        results.append(RagCaseResult(case["id"], case.get("answerable", True),
                                     answered, hit, rr, g, j))

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    summary = {
        "n_cases": len(results),
        "hit_rate": mean([r.hit for r in results if r.hit is not None]),
        "mrr": mean([r.rr for r in results if r.rr is not None]),
        "answer_coverage": mean([1.0 if r.answered else 0.0 for r in answerable]),
        "false_answer_rate": mean([1.0 if r.answered else 0.0 for r in unanswerable]),
        "groundedness": mean([r.groundedness for r in results if r.groundedness is not None]),
        "judge_score": mean([r.judge_score for r in results if r.judge_score is not None]),
        "cases": [asdict(r) for r in results],
    }
    return summary


def threshold_sweep(thresholds: list[float], top_k: int = 5) -> list[dict]:
    """Coverage-vs-safety curve for the similarity threshold.

    The trade-off made measurable: raising the threshold should cut
    false_answer_rate (answers to unanswerable questions) faster than it
    cuts answer_coverage (answers to answerable ones). The report plots
    both so the production threshold becomes a justified choice.
    """
    rows = []
    for t in thresholds:
        s = evaluate_rag(top_k=top_k, threshold=t, use_llm_judge=False)
        rows.append({
            "threshold": t,
            "answer_coverage": s["answer_coverage"],
            "false_answer_rate": s["false_answer_rate"],
            "groundedness": s["groundedness"],
        })
    return rows
