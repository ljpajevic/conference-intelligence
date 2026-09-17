# Evaluation Report — 2026-09-16

Config fingerprint: `8396ae66da55`

## RAG pipeline

| n_cases | hit_rate | mrr | answer_coverage | false_answer_rate | groundedness | judge_score |
|---|---|---|---|---|---|---|
| 25 | 0.000 | 0.000 | 0.750 | 0.200 | 0.793 | 0.000 |


Cases evaluated: 25. `false_answer_rate` = answers produced for questions the corpus cannot answer (lower is better); `answer_coverage` = answerable questions actually answered (higher is better). `hit_rate` and `mrr` read 0 by construction: no case in the RAG golden set carries `relevant_chunk_ids`, so there is nothing for the retrieval-ranking metrics to score.

## Similarity-threshold calibration

| threshold | answer_coverage | false_answer_rate | groundedness |
|---|---|---|---|
| 0.250 | 0.950 | 1.000 | 0.680 |
| 0.400 | 1.000 | 1.000 | 0.638 |
| 0.600 | 0.750 | 0.200 | 0.819 |
| 0.800 | 0.050 | 0.000 | 1.000 |


Production threshold should sit where false answers are suppressed at minimal coverage cost.

## Conference relevance ranking

| paper_weight | cfp_weight | ndcg@3 | precision@3 | cfp_coverage |
|---|---|---|---|---|
| 0.850 | 0.150 | 0.922 | 0.900 | 1.000 |


All conferences scored with the mixed paper + CFP formula.

## Paper/CFP weight sweep

| paper_weight | cfp_weight | ndcg@3 | precision@3 |
|---|---|---|---|
| 0.000 | 1.000 | 0.790 | 0.767 |
| 0.250 | 0.750 | 0.861 | 0.833 |
| 0.500 | 0.500 | 0.902 | 0.867 |
| 0.750 | 0.250 | 0.910 | 0.900 |
| 1.000 | 0.000 | 0.932 | 0.900 |


Production uses 0.85/0.15 — the sweep either validates or improves that choice.
