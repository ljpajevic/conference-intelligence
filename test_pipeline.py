# Runs the full DAG with minimal scope so each node can be verified
# without triggering a full scrape.
#
# Usage:
#   python test_pipeline.py            # runs all checks
#   python test_pipeline.py registry   # runs only the registry node check

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import run_pipeline

# config

TEST_DESCRIPTION = "AI assisted edge offloading"
TEST_CONFERENCES = ["sigcomm", "conext", "imc", "infocom", "mobicom", "eurosys", "mobisys", "icdcs"]
TEST_YEARS       = [2024]

# helpers

PASS = "PASS"
FAIL = "FAIL"

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return condition


# check each agent node

def check_registry(result: dict) -> bool:
    print("\n── Registry Node ──")
    meta = result.get("conference_metadata", {})

    ok = True
    ok &= check("conference_metadata is a dict", isinstance(meta, dict))

    for conf in TEST_CONFERENCES:
        ok &= check(f"{conf} loaded", conf in meta)

    if TEST_CONFERENCES and TEST_CONFERENCES[0] in meta:
        conf = meta[TEST_CONFERENCES[0]]
        ok &= check("full_name present",         bool(conf.get("full_name")))
        ok &= check("topic_areas_seed is list",  isinstance(conf.get("topic_areas_seed"), list))

    ok &= check(
        "conferences_in_scope matches requested",
        set(result.get("conferences_in_scope", [])) == set(TEST_CONFERENCES),
        str(result.get("conferences_in_scope")),
    )
    return ok


def check_paper_scraper(result: dict) -> bool:
    print("\n── Paper Scraper Node ──")
    path_str = result.get("papers_df_path", "")

    ok = True
    ok &= check("papers_df_path is set",    bool(path_str))
    ok &= check("parquet file exists",      Path(path_str).exists() if path_str else False,
                path_str)

    if path_str and Path(path_str).exists():
        try:
            import pandas as pd
            df = pd.read_parquet(path_str)
            ok &= check("parquet has rows",       len(df) > 0,       f"{len(df)} rows")
            ok &= check("conference col present", "conference" in df.columns)
            ok &= check("year col present",       "year"       in df.columns)
            ok &= check("abstract col present",   "abstract"   in df.columns)

            # test coverage as ratio of how many abstracts are present
            n_abs = df["abstract"].notna().sum()
            ok &= check(
                "at least 50% abstracts",
                n_abs / len(df) >= 0.5,
                f"{n_abs}/{len(df)} = {n_abs/len(df):.0%}",
            )

            for conf in TEST_CONFERENCES:
                present = conf in df["conference"].unique()
                ok &= check(f"{conf} papers present", present)

        except Exception as e:
            print(f"  {FAIL}  could not read parquet: {e}")
            ok = False

    return ok


def check_cfp_scraper(result: dict) -> bool:
    print("\n── CFP Scraper Node ──")
    cfp = result.get("cfp_data", {})

    ok = True
    ok &= check("cfp_data is a dict",    isinstance(cfp, dict))

    for conf in TEST_CONFERENCES:
        ok &= check(f"{conf} entry present", conf in cfp)
        if conf in cfp:
            entry = cfp[conf]
            ok &= check(f"{conf} has 'deadlines' key", "deadlines" in entry)
            ok &= check(f"{conf} has 'topics' key",    "topics"    in entry)
            ok &= check(f"{conf} has 'url' key",       "url"       in entry)
            ok &= check(
                f"{conf} deadlines is list",
                isinstance(entry.get("deadlines"), list),
            )
            ok &= check(
                f"{conf} topics is list",
                isinstance(entry.get("topics"), list),
                str(entry.get("topics", [])[:3]),
            )
            deadlines = entry.get("deadlines", [])
            status = PASS if deadlines else "⚠️  WARN"
            print(f"  {status}  {conf} deadlines: {len(deadlines)} found")

    return ok

def check_trend_analysis(result: dict) -> bool:
    print("\n── Trend Analysis Node ──")
    trends = result.get("trends", {})

    ok = True
    ok &= check("trends is a dict",      isinstance(trends, dict))
    for conf in TEST_CONFERENCES:
        ok &= check(f"{conf} entry present", conf in trends)

    for conf in TEST_CONFERENCES:
        if conf not in trends:
            continue
        t = trends[conf]

        ok &= check(f"{conf} has clusters",    isinstance(t.get("clusters"), list))
        ok &= check(f"{conf} has summaries",   bool(t.get("summaries")))
        ok &= check(f"{conf} has trajectory",  isinstance(t.get("trajectory"), dict))

        clusters = t.get("clusters", [])
        ok &= check(f"{conf} at least 1 cluster", len(clusters) > 0, f"{len(clusters)} clusters")

        if clusters:
            c = clusters[0]
            ok &= check(f"{conf} cluster has label",  bool(c.get("label")))
            ok &= check(f"{conf} cluster has size",   isinstance(c.get("size"), int))

        traj = t.get("trajectory", {})
        ok &= check(f"{conf} trajectory has interpretation", bool(traj.get("interpretation")))
        ok &= check(f"{conf} trajectory has counts",         isinstance(traj.get("counts"), dict))

    return ok

def check_relevance(result: dict) -> bool:
    print("\n── Relevance Node ──")
    recs = result.get("recommendations", [])

    ok = True
    ok &= check("recommendations is a list",   isinstance(recs, list))
    ok &= check("at least one recommendation", len(recs) > 0, f"{len(recs)} total")

    if recs:
        top = recs[0]
        ok &= check("top rec has conference", bool(top.get("conference")))
        ok &= check("top rec has score",      isinstance(top.get("score"), (int, float)))
        ok &= check("top rec has rationale",  bool(top.get("rationale")))
        ok &= check("score in 0-10 range",    0 <= top.get("score", -1) <= 10,
                    str(top.get("score")))

        print(f"\n  Top recommendation: {top.get('conference')} "
              f"(score={top.get('score')})")
        print(f"  Rationale: {top.get('rationale', '')[:120]}…")

    return ok

def check_errors(result: dict) -> bool:
    print("\n── Errors ──")
    errors = result.get("errors", [])
    if not errors:
        print(f"  {PASS}  no errors")
        return True
    for e in errors:
        print(f"  ⚠️  WARN  {e}")
    # errors are warnings, not hard failures
    # pipeline can still be useful
    return True

def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print("=" * 52)
    print("PIPELINE TEST")
    print(f"  conferences : {TEST_CONFERENCES}")
    print(f"  years       : {TEST_YEARS}")
    print(f"  description : {TEST_DESCRIPTION}")
    print("=" * 52)

    print("\nRunning pipeline…")
    result = run_pipeline(
        user_research_description=TEST_DESCRIPTION,
        conferences=TEST_CONFERENCES,
        years=TEST_YEARS,
    )

    # run checks either per agent or run the entire pipeline
    results = {}

    if target in ("all", "registry"):
        results["registry"]      = check_registry(result)

    if target in ("all", "paper_scraper", "paper"):
        results["paper_scraper"] = check_paper_scraper(result)

    if target in ("all", "cfp_scraper", "cfp"):
        results["cfp_scraper"]   = check_cfp_scraper(result)

    if target == "all":
        results["errors"]        = check_errors(result)

    if target in ("all", "trend_analysis", "trend"):
        results["trend_analysis"] = check_trend_analysis(result)

    if target in ("all", "relevance"):
        results["relevance"] = check_relevance(result)

    # summary
    print("\n" + "=" * 52)
    print("SUMMARY")
    print("=" * 52)
    all_passed = True
    for node, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {node}")
        all_passed &= passed

    print()
    if all_passed:
        print("All checks passed ✓")
    else:
        print("Some checks failed — see above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
