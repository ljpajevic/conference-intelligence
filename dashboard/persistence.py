import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from datetime import datetime

from config import DATA_DIR

STATE_PATH = DATA_DIR / "cache" / "last_run.pkl"

def save_state(result: dict, inputs: dict) -> None:
    """Persist pipeline result and the inputs that produced it."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result":    result,
        "inputs":    inputs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(STATE_PATH, "wb") as f:
        pickle.dump(payload, f)


def load_state() -> dict | None:
    """Load the most recent pipeline run, or None if no cache exists."""
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [persistence] failed to load state: {e}")
        return None
