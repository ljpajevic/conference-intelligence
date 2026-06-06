import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from core.registry import list_conferences, init_db
from main import run_data_pipeline, run_recommendations

from rag.rag_pipeline import ask, corpus_ready
from rag.retriever import index_exists

from dashboard.persistence import (
    save_data_state, load_data_state,
    save_recommendations, load_recommendations,
)

# page config
st.set_page_config(
    page_title="Conference Intelligence",
    layout="wide",
)

init_db()

st.title("Conference Intelligence")
st.caption(
    "Multi-agent analysis of networking & systems conferences. "
    "Recommends where to submit based on past papers + upcoming CFPs."
)


# Sidebar - inputs

available_confs = [c["name"] for c in list_conferences()]

with st.sidebar:
    st.header("Inputs")

    description = st.text_area(
        "Research description",
        value="scalable edge computing",
        help="One or two sentences describing what you work on.",
        height=100,
    )

    conferences = st.multiselect(
        "Conferences",
        options=available_confs,
        default=available_confs,
    )

    years = st.multiselect(
        "Years (historical papers)",
        options=[2023, 2024, 2025, 2026],
        default=[2023, 2024, 2025],
    )

    st.divider()
    st.subheader("Pipeline")

    col_load, col_data, col_rec = st.columns(3)
    with col_load:
        load_clicked = st.button("Load cached", use_container_width=True)
    with col_data:
        data_clicked = st.button("Refresh data", use_container_width=True)
    with col_rec:
        rec_clicked = st.button("Recommend", type="primary", use_container_width=True)

    st.caption("⚠️ Refresh data scrapes papers and CFPs — takes several minutes and cannot be cancelled from the UI.")


# session state init

for key, default in [
    ("data_result", None),
    ("data_inputs", None),
    ("data_timestamp", None),
    ("rec_result", None),
    ("rec_inputs", None),
    ("rec_timestamp", None),
    ("pipeline_running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# load cached

if load_clicked:
    data_payload = load_data_state()
    rec_payload  = load_recommendations()

    if data_payload:
        st.session_state.data_result    = data_payload["result"]
        st.session_state.data_inputs    = data_payload["inputs"]
        st.session_state.data_timestamp = data_payload["timestamp"]

    if rec_payload:
        st.session_state.rec_result    = rec_payload["result"]
        st.session_state.rec_inputs    = rec_payload["inputs"]
        st.session_state.rec_timestamp = rec_payload["timestamp"]

    if data_payload or rec_payload:
        st.sidebar.success("Cached state loaded.")
    else:
        st.sidebar.warning("No cached run found.")


# refresh data pipeline

if data_clicked and not st.session_state.pipeline_running:
    if not conferences:
        st.sidebar.error("Please select at least one conference.")
    else:
        st.session_state.pipeline_running = True
        with st.spinner("Running data pipeline — scraping papers and CFPs. This takes several minutes."):
            try:
                result = run_data_pipeline(
                    conferences=conferences,
                    years=years,
                )
            finally:
                st.session_state.pipeline_running = False

        inputs = {"conferences": conferences, "years": years}
        save_data_state(result, inputs)
        st.session_state.data_result    = result
        st.session_state.data_inputs    = inputs
        st.session_state.data_timestamp = "just now"
        st.sidebar.success("Data pipeline complete.")


# run recommendations

if rec_clicked and not st.session_state.pipeline_running:
    if not description.strip():
        st.sidebar.error("Please enter a research description.")
    elif st.session_state.data_result is None:
        st.sidebar.error("No data available. Run 'Refresh data' first or load a cached run.")
    else:
        st.session_state.pipeline_running = True
        with st.spinner("Scoring conferences against your research description..."):
            try:
                result = run_recommendations(
                    user_research_description=description,
                    conferences=conferences,
                    papers_df_path=st.session_state.data_result.get("papers_df_path", ""),
                    cfp_data=st.session_state.data_result.get("cfp_data", {}),
                )
            finally:
                st.session_state.pipeline_running = False

        inputs = {"description": description, "conferences": conferences}
        save_recommendations(result, inputs)
        st.session_state.rec_result    = result
        st.session_state.rec_inputs    = inputs
        st.session_state.rec_timestamp = "just now"
        st.sidebar.success("Recommendations complete.")


# sidebar status

with st.sidebar:
    st.divider()
    if st.session_state.data_inputs:
        st.caption("**Data**")
        st.caption(f"From: {st.session_state.data_timestamp}")
        st.caption(
            f"Confs: {len(st.session_state.data_inputs['conferences'])} · "
            f"Years: {st.session_state.data_inputs['years']}"
        )
    if st.session_state.rec_inputs:
        st.caption("**Recommendations**")
        st.caption(f"From: {st.session_state.rec_timestamp}")
        st.caption(f"Description: {st.session_state.rec_inputs['description'][:60]}…")


# empty state

if st.session_state.data_result is None and st.session_state.rec_result is None:
    st.info(
        "No data loaded. "
        "Choose **Load cached** to load the last run, "
        "**Refresh data** to scrape fresh papers and CFPs, "
        "or **Recommend** to score conferences against your research description (requires data)."
    )
    st.stop()


# Tabs

tab_recs, tab_trends, tab_cfp, tab_insights, tab_errors = st.tabs([
    "Recommendations",
    "Trends",
    "CFP Details",
    "Insights",
    "Errors",
])


# Tab 1 - Recommendations

with tab_recs:
    recs = (st.session_state.rec_result or {}).get("recommendations", [])

    if not recs:
        st.info("No recommendations yet. Enter a research description and click **Recommend**.")
    else:
        st.subheader(f"Ranked by relevance — {len(recs)} conferences")
        if st.session_state.rec_inputs:
            st.caption(f"For: {st.session_state.rec_inputs['description']}")

        for i, r in enumerate(recs, start=1):
            with st.container(border=True):
                col_a, col_b = st.columns([3, 2])

                with col_a:
                    st.markdown(f"### {i}. {r['conference'].upper()}")
                    st.write(r.get("rationale", "(no rationale)"))

                with col_b:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Combined", f"{r['score']:.1f}")
                    m2.metric("Paper",    f"{r.get('paper_score', 0):.1f}")
                    m3.metric("CFP",      f"{r.get('cfp_score', 0):.1f}")
                    if not r.get("cfp_available", True):
                        st.caption("⚠️ CFP unavailable — score is paper-only")

                with st.expander("Evidence"):
                    titles     = r.get("top_titles", [])
                    cfp_topics = r.get("top_cfp_topics", [])
                    if titles:
                        st.markdown("**Top matching papers:**")
                        for t in titles:
                            st.markdown(f"- {t}")
                    if cfp_topics:
                        st.markdown("**Top matching CFP topics:**")
                        for t in cfp_topics:
                            st.markdown(f"- {t}")


# Tab 2 - Trends

with tab_trends:
    trends = (st.session_state.data_result or {}).get("trends", {})

    if not trends:
        st.info("No trend data available. Run **Refresh data** to generate trends.")
    else:
        for conf_name in sorted(trends.keys()):
            t = trends[conf_name]
            with st.expander(f"**{conf_name.upper()}**", expanded=False):
                summary = t.get("summaries")
                if summary:
                    st.write(summary)
                else:
                    st.caption("_(no conference summary)_")

                clusters = t.get("clusters", [])
                if clusters:
                    st.markdown("##### Clusters")
                    for c in clusters:
                        with st.container(border=True):
                            cl_col1, cl_col2 = st.columns([3, 1])
                            with cl_col1:
                                st.markdown(f"**{c.get('label', 'Untitled')}**")
                                st.caption(c.get("summary", ""))
                                for tit in c.get("representative_titles", []):
                                    st.markdown(f"- {tit}")
                            with cl_col2:
                                st.metric("Papers", c.get("size", 0))

                traj   = t.get("trajectory", {})
                counts = traj.get("counts", {})
                interpretation = traj.get("interpretation", "")

                if counts:
                    st.markdown("##### Year-over-year")
                    if interpretation:
                        st.write(interpretation)
                    rows = []
                    for label, year_counts in counts.items():
                        for year, n in year_counts.items():
                            rows.append({"year": str(year), "label": label, "count": n})
                    if rows:
                        chart_df = (
                            pd.DataFrame(rows)
                            .pivot(index="year", columns="label", values="count")
                            .fillna(0)
                            .sort_index()
                        )
                        st.line_chart(chart_df)


# Tab 3 - CFP Details

with tab_cfp:
    cfp = (st.session_state.data_result or {}).get("cfp_data", {})

    if not cfp:
        st.info("No CFP data available. Run **Refresh data** to scrape CFPs.")
    else:
        for conf_name in sorted(cfp.keys()):
            entry = cfp[conf_name]
            with st.container(border=True):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"### {conf_name.upper()}")
                    url = entry.get("url")
                    if url:
                        st.caption(f"Source: [{url}]({url})")

                deadlines = entry.get("deadlines", [])
                topics    = entry.get("topics", [])

                with col_b:
                    m1, m2 = st.columns(2)
                    m1.metric("Deadlines", len(deadlines))
                    m2.metric("Topics",    len(topics))

                if deadlines:
                    st.markdown("**Deadlines:**")
                    st.dataframe(pd.DataFrame(deadlines), use_container_width=True, hide_index=True)
                else:
                    st.caption("_(no deadlines extracted)_")

                if topics:
                    st.markdown("**Topics:**")
                    st.write(", ".join(topics))
                else:
                    st.caption("_(no topics extracted)_")

# Tab 4 - Insights

with tab_insights:
    if not corpus_ready():
        st.warning("No corpus available. Run **Refresh data** first to build the paper index.")
    else:
        if not index_exists():
            st.info("Index will be built on your first query. This takes a minute.")

        query = st.text_input(
            "Ask a question about the research corpus",
            placeholder="e.g. What approaches exist for offloading compute to the edge?",
        )

        with st.expander("Filters (optional)"):
            filter_confs = st.multiselect(
                "Conferences", options=available_confs, default=[]
            )
            filter_years = st.multiselect(
                "Years", options=[2023, 2024, 2025], default=[]
            )

        ask_clicked = st.button("Ask", type="primary")

        if ask_clicked:
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating answer..."):
                    result = ask(
                        query=query,
                        conferences=filter_confs or None,
                        years=filter_years or None,
                    )

                if not result["grounded"]:
                    st.warning("No closely related papers found in the corpus for this question.")
                else:
                    st.markdown("### Answer")
                    st.write(result["answer"])

                    sources = result["sources"]
                    if sources:
                        st.markdown("### Sources")
                        for s in sources:
                            doi_link = f" — [DOI]({s['doi']})" if s["doi"] else ""
                            st.markdown(
                                f"- **{s['paper_title']}** "
                                f"({s['conference'].upper()} {s['year']}){doi_link}"
                            )

                    with st.expander("Retrieved chunks"):
                        for chunk in result["chunks"]:
                            st.markdown(
                                f"**{chunk['paper_title']}** "
                                f"({chunk['conference'].upper()} {chunk['year']}) "
                                f"— similarity: {chunk['similarity']}"
                            )
                            st.caption(chunk["text"])
                            st.divider()

# Tab 5 - Errors

with tab_errors:
    data_errors = (st.session_state.data_result or {}).get("errors", [])
    rec_errors  = (st.session_state.rec_result or {}).get("errors", [])
    errors = data_errors + rec_errors

    if not errors:
        st.success("No errors in the last run.")
    else:
        st.warning(f"{len(errors)} error(s):")
        for e in errors:
            st.code(e, language=None)
