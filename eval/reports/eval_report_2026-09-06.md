# Evaluation Report — 2026-09-06

Config fingerprint: `8396ae66da55`

## RAG pipeline

| n_cases | hit_rate | mrr | answer_coverage | false_answer_rate | groundedness | judge_score |
|---|---|---|---|---|---|---|
| 25 | 0.000 | 0.000 | 0.750 | 0.200 | 0.826 | 0.000 |


Cases evaluated: 25. `false_answer_rate` = answers produced for questions the corpus cannot answer (lower is better); `answer_coverage` = answerable questions actually answered (higher is better). `hit_rate` and `mrr` read 0 by construction: the RAG golden set carries no `relevant_chunk_ids`, so there is nothing for the retrieval-ranking metrics to score.

## Conference relevance ranking

| paper_weight | cfp_weight | ndcg@3 | precision@3 | cfp_coverage |
|---|---|---|---|---|
| 0.850 | 0.150 | 0.906 | 0.867 | 1.000 |


All conferences scored with the mixed paper + CFP formula.
