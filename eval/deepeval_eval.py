"""DeepEval metrics for the RAG pipeline.

Runs faithfulness, answer relevancy, and contextual recall against the same
golden set used by rag_eval.py. Results are complementary to the hand-rolled
metrics: faithfulness catches contradictions, answer relevancy catches
topic drift, contextual recall measures how much of the reference answer
the retrieved chunks actually support.

Requires:
    DEEPEVAL_API_KEY environment variable (free tier sufficient).
    pip install deepeval==4.1.1
"""
from dataclasses import dataclass, asdict
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase

from . import adapters
from .rag_eval import load_golden
from .metrics import mean


@dataclass
class DeepEvalCaseResult:
    case_id: str
    answerable: bool
    answered: bool
    faithfulness: float | None
    answer_relevancy: float | None
    contextual_recall: float | None


def run_deepeval(top_k: int = 5, threshold: float | None = None) -> dict:
    """Run DeepEval metrics over the RAG golden set.

    Returns a summary dict compatible with report.py.
    Skips unanswerable cases where generate_answer returns None.
    """
    golden = load_golden()
    results: list[DeepEvalCaseResult] = []

    faithfulness_metric     = FaithfulnessMetric(threshold=0.5, verbose_mode=False)
    relevancy_metric        = AnswerRelevancyMetric(threshold=0.5, verbose_mode=False)
    recall_metric           = ContextualRecallMetric(threshold=0.5, verbose_mode=False)

    for case in golden:
        chunks  = adapters.apply_threshold(
            adapters.retrieve(case["question"], top_k=top_k), threshold)
        answer  = adapters.generate_answer(case["question"], chunks, threshold=threshold)
        answered = answer is not None

        if not answered:
            results.append(DeepEvalCaseResult(
                case_id=case["id"],
                answerable=case.get("answerable", True),
                answered=False,
                faithfulness=None,
                answer_relevancy=None,
                contextual_recall=None,
            ))
            continue

        retrieval_context = [c.text for c in chunks]
        reference = case.get("reference_answer", "")

        test_case = LLMTestCase(
            input=case["question"],
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=reference if reference else None,
        )

        # run metrics individually to handle missing reference_answer gracefully
        f_score = _safe_score(faithfulness_metric, test_case)
        r_score = _safe_score(relevancy_metric, test_case)
        cr_score = _safe_score(recall_metric, test_case) if reference else None

        results.append(DeepEvalCaseResult(
            case_id=case["id"],
            answerable=case.get("answerable", True),
            answered=True,
            faithfulness=f_score,
            answer_relevancy=r_score,
            contextual_recall=cr_score,
        ))

    answered_results = [r for r in results if r.answered]

    summary = {
        "n_cases":            len(results),
        "n_answered":         len(answered_results),
        "faithfulness":       mean([r.faithfulness for r in answered_results
                                    if r.faithfulness is not None]),
        "answer_relevancy":   mean([r.answer_relevancy for r in answered_results
                                    if r.answer_relevancy is not None]),
        "contextual_recall":  mean([r.contextual_recall for r in answered_results
                                    if r.contextual_recall is not None]),
        "cases":              [asdict(r) for r in results],
    }
    return summary


def _safe_score(metric, test_case: LLMTestCase) -> float | None:
    """Run a single DeepEval metric and return its score, or None on failure."""
    try:
        metric.measure(test_case)
        return metric.score
    except Exception as e:
        print(f"  [deepeval] {metric.__class__.__name__} failed: {e}")
        return None
