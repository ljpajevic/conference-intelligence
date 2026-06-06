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
Ranks conferences against the user's research description using deterministic cosine similarity over papers and CFP topics. CFP-weighted scoring `(0.3 × paper + 0.7 × CFP)`. LLM: Ollama for rationale only.

#### Stack

| Layer | Choice |
|---|---|
| Agent framework | LangGraph 1.2.1 |
| LLM (quality calls) | Groq `llama-3.3-70b-versatile` |
| LLM (quota-free calls) | Ollama `llama3.1:8b` |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Paper metadata | DBLP XML |
| Abstract enrichment | OpenAlex API |
| Registry storage | SQLite |
| Data storage | Parquet |
| Caching | File-based JSON (`core/cache.py`) |
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

<img src="assets/dashboard_start.png" alt="Dashboard - Start" width="600">

#### Dashboard

The dashboard has four tabs: Recommendations, Trends, CFP Details, and Errors.

**Recommendations** — highest-ranked conference matches with rationale and matching CFP topics. Re-run with any research description without touching the underlying data.

**Trends** — most popular themes per conference, year over year.

**CFP Details** — trending topics and submission deadlines per conference.

**Errors** — any errors accumulated during the pipeline run, such as failed CFP fetches or missing abstracts.

<img src="assets/dashboard_tabs.png" alt="Dashboard Tabs" width="600">

#### Known Limitations

- USENIX-published venues (NSDI, OSDI, USENIX Security) not supported due to insufficient abstract coverage across enrichment sources
- Poster and demo filtering uses a 4-page threshold; would need recalibration for venues with very short full papers
- **Refresh data** cannot be cancelled mid-execution from the UI; kill the process from the terminal if needed
- Cache check in `paper_agent` triggers a full re-scrape if any `(conference, year)` combination is missing; missing slices are not fetched incrementally
