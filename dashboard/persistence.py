import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from datetime import datetime
from config import DATA_DIR

DATA_STATE_PATH = DATA_DIR / "cache" / "last_data_run.pkl"
RECOMMENDATIONS_PATH = DATA_DIR / "cache" / "last_recommendations.pkl"


def save_data_state(result: dict, inputs: dict) -> None:
    """Persist data pipeline result (papers, CFP, trends) and the inputs that produced it."""
    DATA_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result":    result,
        "inputs":    inputs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(DATA_STATE_PATH, "wb") as f:
        pickle.dump(payload, f)


def load_data_state() -> dict | None:
    """Load the most recent data pipeline run, or None if no cache exists."""
    if not DATA_STATE_PATH.exists():
        return None
    try:
        with open(DATA_STATE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [persistence] failed to load data state: {e}")
        return None


def save_recommendations(result: dict, inputs: dict) -> None:
    """Persist recommendations result and the inputs that produced it."""
    RECOMMENDATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result":    result,
        "inputs":    inputs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(RECOMMENDATIONS_PATH, "wb") as f:
        pickle.dump(payload, f)


def load_recommendations() -> dict | None:
    """Load the most recent recommendations run, or None if no cache exists."""
    if not RECOMMENDATIONS_PATH.exists():
        return None
    try:
        with open(RECOMMENDATIONS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [persistence] failed to load recommendations: {e}")
        return None
