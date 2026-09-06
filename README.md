![CI](https://github.com/ljpajevic/conference-intelligence/actions/workflows/ci.yml/badge.svg)


### Overview

Multi-agent research conference intelligence and recommendation system.

The user provides a research description and the system scrapes paper metadata from academic conferences, discovers trends, fetches CFP deadlines, and recommends the best conferences to submit to. Built with LangGraph.

Currently supported conferences: SIGCOMM, CoNEXT, IMC, MobiSys, MobiCom, EuroSys (ACM), INFOCOM, ICDCS (IEEE). Papers covered: 2023–2025.

#### Architecture

The system runs as two independent graphs sharing a common data layer.

**Data graph** — run rarely, when you want fresh papers and CFPs:
```
registry → paper_scraper → cfp_scraper → trend_analysis → END
```

**Query graph** — run on demand, once per research description:
```
registry → relevance → END
```

The query graph reads from parquet and CFP cache produced by the data graph. No scraping happens during recommendations.

##### registry_agent.py
First node in both graphs. Resolves conference scope into fully populated metadata (URLs, DBLP keys, CFP URLs). Also exposes a conversational CLI for manual registry management. LLM: Ollama.

##### paper_agent.py
Scrapes paper metadata from DBLP, enriches abstracts via OpenAlex, computes and persists embeddings to parquet. No LLM. Smart cache per `(conference, year)` to avoid full re-scrape when only new conferences or years are added.

##### cfp_agent.py
Fetches and parses CFP pages per conference, extracts deadlines and topic areas. Registry-driven URL resolution with year substitution. Manual override path for JS-rendered pages. LLM: Groq.

##### trend_agent.py
Clusters papers per conference using KMeans, labels clusters and writes conference summaries (Groq), and interprets year-over-year trajectory (Ollama). Separates deterministic clustering from LLM-generated descriptions.

##### relevance_agent.py
Ranks conferences against the user's research description using deterministic cosine similarity over papers and CFP topics. Final score is `α × paper + (1-α) × CFP`, with `ALPHA = 0.85` (see Evaluation). Both components are size-normalised: the paper score is the mean of each venue's top-decile similarities rather than a fixed top-k, and the CFP score is the mean of the top 3 topic similarities rather than a single max. LLM: Ollama is for rationale only, doesn't enter the score.

#### Stack

| Layer | Choice |
|---|---|
| Agent framework | LangGraph 1.2.1 |
| LLM (quality calls) | Groq `openai/gpt-oss-120b` |
| LLM (quota-free calls) | Ollama `llama3.1:8b` |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Paper metadata | DBLP XML |
| Abstract enrichment | OpenAlex API |
| Registry storage | SQLite |
| Data storage | Parquet |
| Caching | File-based JSON (`core/cache.py`) |
| Vector store | ChromaDB (persistent, local) |
| Runtime | Python 3.11+ |

#### Quick Start

**Prerequisites:**
- Groq API key set as `GROQ_API_KEY` environment variable
- Ollama running locally with `llama3.1:8b` pulled
- `pip install -r requirements.txt`

```
streamlit run dashboard/app.py
```

On first launch the registry is initialized automatically. Select conferences and years, then:

1. Press **Refresh data** to scrape papers and CFPs. This takes several minutes and only needs to be repeated when you want fresher data.
2. Enter a research description and press **Recommend** to score conferences against it. This is fast and can be re-run with different descriptions without re-scraping.

<img src="assets/dashboard_recommendations.png" alt="Dashboard - Start" width="600">

#### Dashboard

The dashboard has five tabs: Recommendations, Trends, CFP Details, Insights, and Errors.

**Recommendations** — highest-ranked conference matches with rationale and matching CFP topics. Re-run with any research description without touching the underlying data.

**Trends** — most popular themes per conference, year over year.

**CFP Details** — trending topics and submission deadlines per conference.

**Insights** - RAG-powered querying of the paper corpus, with strictly grounded responses and explicit no-result handling.

**Errors** — any errors accumulated during the pipeline run, such as failed CFP fetches or missing abstracts.

<img src="assets/dashboard_insights.png" alt="Dashboard Tabs - Insights" width="600">

#### Known Limitations

- USENIX-published venues (NSDI, OSDI, USENIX Security) not supported due to insufficient abstract coverage across enrichment sources.
- Poster and demo filtering uses a 4-page threshold; would need recalibration for venues with very short full papers.
- **Refresh data** cannot be cancelled mid-execution from the UI; kill the process from the terminal if needed.
- Cache check in `paper_agent` triggers a full re-scrape if any `(conference, year)` combination is missing; missing slices are not fetched incrementally.
- Golden sets are small (10 relevance cases, 25 RAG cases) and graded by a single annotator. Swapping 5 cases for 10 moved NDCG@3 by ~0.05 on identical code, which is larger than most differences the weight sweep resolves; treat individual sweep points as indicative, not decisive.
- Relevance grades are static, so the evaluation cannot reward the one thing CFP topics uniquely provide: what a venue wants *next* year. A venue that has shifted scope shows up in its CFP before it shows up in its published papers.
- Venues list 10–22 CFP topics, and a top-k statistic over so few still mildly favours the longer lists (~0.04 on synthetic data). Too narrow a range for a quantile to fix; documented, not solved.
- `MIN_K = 10` in the paper score reintroduces a small size bias for venues under ~100 papers. CoNEXT, the smallest at 106, sits just above that boundary
Keep the four existing bullets above these. Delete the old "no warning this
happened" bullet entirely — the dashboard flags it and eval reports carry
`cfp_coverage`.


#### Evaluation

Hand-rolled harness (`eval/`), with [DeepEval](https://github.com/confident-ai/deepeval) wired in for standard LLM metrics but not yet run.
Two golden sets: 25 RAG cases (20 answerable, 5 unanswerable) and 10 expert-graded relevance cases (single annotator).

**RAG pipeline** (threshold=0.60, top-k=5):

| Metric | Value |
|---|---|
| answer_coverage | 0.750 |
| false_answer_rate | 0.200 |
| groundedness | 0.826 |

`hit_rate` and `mrr` read 0 because the RAG golden set has no `relevant_chunk_ids` yet (populating them is open work).

**Conference relevance ranking** (paper_weight=0.85, cfp_weight=0.15):

| Metric | Value |
|---|---|
| NDCG@3 | 0.906 |
| Precision@3 | 0.867 |

**Corpus-size bias in the paper score.** The score was the mean of a venue's top-20 paper similarities, but a fixed count is a variable quantile. For the largest venue that's the top 2%, for the smallest the top 19%, so large venues got a bonus unrelated to fit. `scripts/check_size_bias.py` measures it by capping every venue to the same paper count. Measured on the eight-venue corpus, the largest held a top-3 slot in 9 of 10 golden cases and fell to 2.3 when capped. After switching to a fixed quantile (`TOP_Q = 0.10`) it reads 2 uncapped against 2.7 capped.


**CFP topics add no measurable ranking value.** An 11-point weight sweep rises monotonically toward paper-only (0.834 at CFP-only against 0.932 at paper-only) and that held for two different CFP statistics, thus seems to be a property of the signal rather than the estimator. `ALPHA` is 0.85 rather than 1.0 deliberately: the gap is within the ~0.05 the golden set moves on its own, and CFP still supplies the rationale text and the dashboard's topic matches. It's also the only forward-looking signal here, which the golden set can't reward (due to static grades).


##### Known gaps in the harness

- The threshold sweep is inert at or below 0.6: `retrieve()` already filters at `MIN_SIMILARITY`, so lower sweep points change nothing. Needs `MIN_SIMILARITY` parameterised through the retrieval call.
- DeepEval is wired in but has never been run.

Reports in [`eval/reports/`](eval/reports/). Each carries a config fingerprint covering the embedding model, `top_k`, `ALPHA` and CFP coverage, so runs made under different scoring do not silently compare. Ranking numbers reproduce exactly across both pinned and older pandas/numpy versions.
