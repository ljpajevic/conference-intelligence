"""FastAPI service for the query path.

Exposes the fast graph (scoring) and the RAG pipeline over HTTP. The data
graph is NOT exposed: scraping takes minutes and is an explicit, local,
opt-in run. This service only reads artifacts a prior data run produced.

Deliberately split from the Streamlit dashboard, which stays the local
operator interface. Run:

    uvicorn api.app:app --port 8000

Startup resolves every artifact the endpoints need and fails loudly if one
is missing, so a misconfigured container dies at boot instead of serving
errors or silently rebuilding a 3k-chunk index on the first request.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

_ctx: dict = {}


# startup

def _resolve_context() -> dict:
    """Locate the artifacts the query path depends on, or raise.

    Every one of these is produced by a data-pipeline run and is gitignored,
    so in a container they must have been baked into the image. Checking at
    startup turns three different runtime failure modes into one boot failure
    with a readable message.
    """
    from config import DATA_DIR
    from core.registry import init_db, list_conferences
    from rag.retriever import index_exists, chunks_parquet_exists

    # 1. registry: cheap, rebuilt from SEED_DATA in code, so just ensure it
    init_db()
    records = list_conferences()
    conferences = [c["name"] for c in records]
    display = {c["name"]: c["full_name"] for c in records}
    if not conferences:
        raise RuntimeError("registry initialised but empty — check SEED_DATA")

    # 2. paper corpus
    parquet = DATA_DIR / "enriched" / "networking_papers_enriched.parquet"
    if not parquet.exists():
        raise RuntimeError(
            f"paper corpus not found at {parquet}. Run the data pipeline "
            f"(python pipeline/paper_pipeline.py) or bake the parquet into "
            f"the image."
        )

    # 3. CFP topics. The tracked JSON export is preferred: it survives a fresh
    #    clone and an image build, and does not drag the Streamlit layer into
    #    the service. The dashboard pickle stays as a fallback so local runs
    #    work unchanged before anyone has exported. Soft failure either way --
    #    without CFP topics the service still ranks on the paper signal, and
    #    /api/health reports which source was used.
    cfp_data, cfp_source, cfp_generated = {}, "none", None
    export_path = DATA_DIR / "cfp_export.json"
    if export_path.exists():
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        cfp_data = payload.get("venues", {})
        cfp_source, cfp_generated = "export", payload.get("generated")
    else:
        from dashboard.persistence import load_data_state
        state = load_data_state()
        if state:
            cfp_data = state["result"].get("cfp_data", {})
            cfp_source, cfp_generated = "pickle", state.get("timestamp")

    # 4. RAG index. Never let _get_collection() rebuild here: it would
    #    re-embed every chunk on a user's first question.
    rag_ready = index_exists()
    if not rag_ready and not chunks_parquet_exists():
        raise RuntimeError(
            "no Chroma index and no chunks parquet — /api/ask cannot work. "
            "Build the index locally and bake data/chroma into the image."
        )
    if not rag_ready:
        raise RuntimeError(
            "chunks parquet present but Chroma index missing. Building it "
            "here would re-embed the whole corpus on the first request; "
            "build it locally and bake data/chroma into the image."
        )

    # 5. embedding model: pay the load cost at boot, not on request one
    from core.tools import get_embedding_model
    get_embedding_model()

    return {
        "conferences": conferences,
        "display": display,
        "papers_df_path": str(parquet),
        "cfp_data": cfp_data,
        "cfp_venues": sorted(cfp_data.keys()),
        "cfp_source": cfp_source,
        "cfp_generated": cfp_generated,
        "rag_ready": rag_ready,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ctx.update(_resolve_context())
    print(f"[api] ready — {len(_ctx['conferences'])} conferences, "
          f"CFP topics for {len(_ctx['cfp_venues'])}")
    yield
    _ctx.clear()


app = FastAPI(
    title="Conference Intelligence API",
    description="Venue recommendations and grounded corpus Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)


# request / response models

class RecommendRequest(BaseModel):
    research_description: str = Field(min_length=10, max_length=2000)
    conferences: list[str] | None = None


class Recommendation(BaseModel):
    conference: str
    full_name: str | None = None
    score: float
    paper_score: float
    cfp_score: float
    mean_score: float
    cfp_available: bool
    top_titles: list[str] = []
    top_cfp_topics: list[str] = []


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    errors: list[str] = []


class RationaleRequest(BaseModel):
    """The client sends back one recommendation it received from /recommend.

    Rationales are generated one venue at a time, on click: eight up front is
    eight LLM calls for text that mostly never gets read.
    """
    research_description: str = Field(min_length=10, max_length=2000)
    recommendation: Recommendation


class RationaleResponse(BaseModel):
    conference: str
    rationale: str


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)
    conferences: list[str] | None = None
    years: list[int] | None = None


class Source(BaseModel):
    paper_title: str
    conference: str
    year: str | int
    doi: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    grounded: bool


# helpers

def _validate_conferences(requested: list[str] | None) -> list[str]:
    known = _ctx["conferences"]
    if not requested:
        return known
    unknown = [c for c in requested if c.lower() not in known]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown conference(s): {', '.join(unknown)}. "
                   f"Known: {', '.join(known)}",
        )
    return [c.lower() for c in requested]


# endpoints

@app.get("/api/health")
def health() -> dict:
    if not _ctx:
        raise HTTPException(status_code=503, detail="service not ready")
    return {
        "status": "ok",
        "conferences": _ctx["conferences"],
        "display": _ctx["display"],
        "cfp_venues": _ctx["cfp_venues"],
        "cfp_source": _ctx["cfp_source"],
        "cfp_generated": _ctx["cfp_generated"],
        "rag_ready": _ctx["rag_ready"],
    }


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    """Rank venues against a research description.

    generate_rationales=False: scoring is deterministic cosine work and
    should stay fast. Rationales come from /api/rationale on demand.
    """
    from main import run_recommendations

    conferences = _validate_conferences(req.conferences)
    try:
        result = run_recommendations(
            user_research_description=req.research_description,
            conferences=conferences,
            papers_df_path=_ctx["papers_df_path"],
            cfp_data=_ctx["cfp_data"],
            generate_rationales=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scoring failed: {e}") from e

    recs = result.get("recommendations", [])
    if not recs:
        raise HTTPException(
            status_code=500,
            detail=f"no venues scored; errors: {result.get('errors', [])}",
        )
    return RecommendResponse(
        recommendations=[
            Recommendation(**r, full_name=_ctx["display"].get(r["conference"]))
            for r in recs
        ],
        errors=result.get("errors", []),
    )


@app.post("/api/rationale", response_model=RationaleResponse)
def rationale(req: RationaleRequest) -> RationaleResponse:
    """Generate the rationale for a single already-scored venue."""
    from agents.relevance_agent import rationale_for, RationaleError

    conf = req.recommendation.conference.lower()
    if conf not in _ctx["conferences"]:
        raise HTTPException(status_code=400, detail=f"unknown conference: {conf}")

    # pass the venue's CFP topics so per-topic similarities are real rather
    # than absent — rec carries only the aggregate cfp_score
    cfp_topics = _ctx["cfp_data"].get(conf, {}).get("topics", []) or None

    try:
        text = rationale_for(
            req.recommendation.model_dump(),
            req.research_description,
            cfp_topics=cfp_topics,
        )
    except RationaleError as e:
        raise HTTPException(status_code=502, detail=f"rationale unavailable: {e}") from e

    return RationaleResponse(conference=conf, rationale=text)


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Grounded Q&A over the paper corpus."""
    from rag.rag_pipeline import ask as rag_ask

    conferences = _validate_conferences(req.conferences) if req.conferences else None
    try:
        result = rag_ask(
            query=req.question,
            top_k=req.top_k,
            conferences=conferences,
            years=req.years,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}") from e

    return AskResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result.get("sources", [])],
        grounded=result["grounded"],
    )


# static frontend, mounted only if built

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
