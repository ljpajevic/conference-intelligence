"""Export CFP data from the last data-pipeline run to a tracked JSON file.

CFP topics feed the ranking's second signal, but they only ever existed inside
data/cache/last_data_run.pkl -- written by the Streamlit dashboard, ignored by
git (data/ and *.pkl both), and therefore absent from a fresh clone and from
any container image. Three consequences this fixes:

  - nobody can reproduce the eval's cfp_coverage 1.0 from the public repo
  - the deployed service would import the dashboard layer to read a pickle
  - the image would have to bake an opaque binary whose format is tied to the
    pandas and Python versions that wrote it

JSON is diffable, reviewable, and loads without the dashboard. The file is a
snapshot: topics and deadlines move with each conference cycle, so re-run this
after every data-pipeline run. The `generated` field is there to make a stale
export obvious rather than silent.

    python scripts/export_cfp.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime, timezone

from config import DATA_DIR

EXPORT_PATH = DATA_DIR / "cfp_export.json"


def build_export() -> dict:
    from dashboard.persistence import load_data_state

    state = load_data_state()
    if state is None:
        raise SystemExit(
            "No data-pipeline run found at data/cache/last_data_run.pkl.\n"
            "Run the data pipeline from the dashboard first "
            "(streamlit run dashboard/app.py -> Refresh data)."
        )

    cfp_data = state["result"].get("cfp_data", {})
    if not cfp_data:
        raise SystemExit(
            "The last data-pipeline run produced no CFP data. Nothing to export."
        )

    venues = {}
    for name in sorted(cfp_data):
        entry = cfp_data[name] or {}
        venues[name] = {
            "topics":    list(entry.get("topics", [])),
            "deadlines": list(entry.get("deadlines", [])),
            "url":       entry.get("url"),
        }
    run = state.get("timestamp")
    if run:
        run = datetime.fromisoformat(run).astimezone(timezone.utc).isoformat(timespec="seconds")

    return {
        "generated":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_run": run,
        "venues":       venues,
    }


def main() -> None:
    export = build_export()
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys so re-exporting unchanged data produces no diff
    EXPORT_PATH.write_text(
        json.dumps(export, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    venues = export["venues"]
    empty = [n for n, v in venues.items() if not v["topics"]]
    print(f"Wrote {EXPORT_PATH.relative_to(Path(__file__).parent.parent)}")
    print(f"  venues:        {len(venues)}")
    print(f"  with topics:   {len(venues) - len(empty)}")
    print(f"  total topics:  {sum(len(v['topics']) for v in venues.values())}")
    print(f"  pipeline run:  {export['pipeline_run']}")
    if empty:
        print(f"  NO topics for: {', '.join(empty)} — these rank paper-only")


if __name__ == "__main__":
    main()
