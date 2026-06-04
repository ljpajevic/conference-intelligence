import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from core.registry import list_conferences
from main import run_pipeline
from dashboard.persistence import save_state, load_state

# page config
st.set_page_config(
    page_title="Conference Intelligence",
    layout="wide",
)

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

    col1, col2 = st.columns(2)
    with col1:
        load_clicked = st.button("Load cached", use_container_width=True)
    with col2:
        run_clicked = st.button("Run fresh", type="primary", use_container_width=True)
    st.caption("⚠️ Fresh runs cannot be cancelled mid-pipeline. They take a few minutes.")


# pipeline execution/state loading

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.inputs = None
    st.session_state.timestamp = None

if load_clicked:
    payload = load_state()
    if payload is None:
        st.sidebar.warning("No cached run found.")
    else:
        st.session_state.result    = payload["result"]
        st.session_state.inputs    = payload["inputs"]
        st.session_state.timestamp = payload["timestamp"]
        st.sidebar.success(f"Loaded cached run from {payload['timestamp']}")

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

if run_clicked and not st.session_state.pipeline_running:
    if not description.strip():
        st.sidebar.error("Please enter a research description.")
    elif not conferences:
        st.sidebar.error("Please select at least one conference.")
    else:
        st.session_state.pipeline_running = True
        with st.spinner("Running pipeline — once started, this cannot be cancelled from the UI."):
            try:
                result = run_pipeline(
                    user_research_description=description,
                    conferences=conferences,
                    years=years,
                )
            finally:
                st.session_state.pipeline_running = False
        inputs = {
            "description":  description,
            "conferences":  conferences,
            "years":        years,
        }
        save_state(result, inputs)
        st.session_state.result    = result
        st.session_state.inputs    = inputs
        st.session_state.timestamp = "just now"
        st.sidebar.success("Pipeline complete.")

# show current input in sidebar
if st.session_state.inputs:
    with st.sidebar:
        st.divider()
        st.caption("**Loaded run**")
        st.caption(f"From: {st.session_state.timestamp}")
        st.caption(f"Description: {st.session_state.inputs['description'][:80]}…")
        st.caption(
            f"Confs: {len(st.session_state.inputs['conferences'])} · "
            f"Years: {st.session_state.inputs['years']}"
        )


# empty state

if st.session_state.result is None:
    st.info(
        "No pipeline results loaded yet. "
        "Choose **Load cached** to load the last run, "
        "or **Run fresh** to start a new analysis."
    )
    st.stop()


result = st.session_state.result


# Tabs

tab_recs, tab_trends, tab_cfp, tab_errors = st.tabs([
    "Recommendations",
    "Trends",
    "CFP Details",
    "Errors",
])


# Tab 1 - Recommendations

with tab_recs:
    recs = result.get("recommendations", [])

    if not recs:
        st.warning("No recommendations available.")
    else:
        st.subheader(f"Ranked by relevance — {len(recs)} conferences")

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
                    titles = r.get("top_titles", [])
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
    trends = result.get("trends", {})

    if not trends:
        st.warning("No trend data available.")
    else:
        for conf_name in sorted(trends.keys()):
            t = trends[conf_name]
            with st.expander(f"**{conf_name.upper()}**", expanded=False):
                summary = t.get("summaries")
                if summary:
                    st.write(summary)
                else:
                    st.caption("_(no conference summary)_")

                # clusters
                clusters = t.get("clusters", [])
                if clusters:
                    st.markdown("##### Clusters")
                    for c in clusters:
                        with st.container(border=True):
                            cl_col1, cl_col2 = st.columns([3, 1])
                            with cl_col1:
                                st.markdown(f"**{c.get('label', 'Untitled')}**")
                                st.caption(c.get("summary", ""))
                                titles = c.get("representative_titles", [])
                                if titles:
                                    for tit in titles:
                                        st.markdown(f"- {tit}")
                            with cl_col2:
                                st.metric("Papers", c.get("size", 0))

                # trajectory
                traj = t.get("trajectory", {})
                counts = traj.get("counts", {})
                interpretation = traj.get("interpretation", "")

                if counts:
                    st.markdown("##### Year-over-year")
                    if interpretation:
                        st.write(interpretation)

                    # build a DataFrame for st.line_chart
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


# Tab 3: CFP Details

with tab_cfp:
    cfp = result.get("cfp_data", {})

    if not cfp:
        st.warning("No CFP data available.")
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
                    df = pd.DataFrame(deadlines)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("_(no deadlines extracted)_")

                if topics:
                    st.markdown("**Topics:**")
                    # Render as a wrapping list
                    st.write(", ".join(topics))
                else:
                    st.caption("_(no topics extracted)_")


# Tab 4: Errors

with tab_errors:
    errors = result.get("errors", [])
    if not errors:
        st.success("No errors in this run.")
    else:
        st.warning(f"{len(errors)} error(s) during pipeline execution:")
        for e in errors:
            st.code(e, language=None)
