"""Run the evaluation suite:  python -m eval.run_eval [--judge] [--quick] [--no-deepeval]
--quick       skips the sweeps (fast inner-loop while developing)
--judge       adds the Ollama LLM-judge groundedness pass (slower, local-only)
--no-deepeval skips DeepEval metrics (e.g. when DEEPEVAL_API_KEY is not set)
"""
import argparse
from . import rag_eval, relevance_eval, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge",       action="store_true", help="use Ollama LLM judge")
    ap.add_argument("--quick",       action="store_true", help="skip sweeps")
    ap.add_argument("--no-deepeval", action="store_true", help="skip DeepEval metrics")
    ap.add_argument("--top-k",       type=int, default=5)
    args = ap.parse_args()

    print("RAG evaluation ...")
    rag = rag_eval.evaluate_rag(top_k=args.top_k, use_llm_judge=args.judge)
    print(f"  hit_rate={rag['hit_rate']:.3f}  mrr={rag['mrr']:.3f}  "
          f"coverage={rag['answer_coverage']:.3f}  "
          f"false_answers={rag['false_answer_rate']:.3f}  "
          f"groundedness={rag['groundedness']:.3f}")

    deepeval_summary = None
    if not args.no_deepeval:
        import os
        if not os.getenv("DEEPEVAL_API_KEY"):
            print("DeepEval skipped — DEEPEVAL_API_KEY not set (use --no-deepeval to suppress this warning)")
        else:
            print("DeepEval metrics ...")
            from . import deepeval_eval
            deepeval_summary = deepeval_eval.run_deepeval(top_k=args.top_k)
            print(f"  faithfulness={deepeval_summary['faithfulness']:.3f}  "
                  f"relevancy={deepeval_summary['answer_relevancy']:.3f}  "
                  f"contextual_recall={deepeval_summary['contextual_recall']:.3f}")

    thresholds = None
    weights    = None
    if not args.quick:
        print("Threshold sweep ...")
        thresholds = rag_eval.threshold_sweep([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        print("Weight sweep ...")
        weights = relevance_eval.weight_sweep()

    print("Relevance ranking evaluation ...")
    ranking = relevance_eval.evaluate_ranking()
    print(f"  ndcg@3={ranking['ndcg@3']:.3f}  precision@3={ranking['precision@3']:.3f}")

    path = report.write_report(
        rag, thresholds, ranking, weights,
        deepeval_summary=deepeval_summary,
        config_extra={"top_k": args.top_k},
    )
    print(f"\nReport written: {path}")


if __name__ == "__main__":
    main()
