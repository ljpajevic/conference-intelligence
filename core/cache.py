# core/cache.py
"""
Simple file-based cache for expensive deterministic operations
(LLM calls, embeddings, etc.) keyed by content hash.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import hashlib
from typing import Any

from config import DATA_DIR


CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def hash_content(content: Any) -> str:
    """
    Stable content hash. Accepts strings, lists, dicts, etc.
    Returns first 16 hex chars (64 bits, enough for collision avoidance).
    """
    if isinstance(content, str):
        payload = content
    else:
        payload = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(namespace: str, key: str) -> Path:
    """e.g. namespace='trend_labels', key='sigcomm_a1b2c3...' → data/cache/trend_labels/sigcomm_a1b2c3.json"""
    ns_dir = CACHE_DIR / namespace
    ns_dir.mkdir(parents=True, exist_ok=True)
    return ns_dir / f"{key}.json"


def cache_get(namespace: str, key: str) -> Any | None:
    """Return cached value or None if not present."""
    path = cache_path(namespace, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [cache] failed to read {path}: {e}")
        return None


def cache_set(namespace: str, key: str, value: Any) -> None:
    """Store value as JSON. Overwrites silently."""
    path = cache_path(namespace, key)
    try:
        path.write_text(json.dumps(value, default=str))
    except Exception as e:
        print(f"  [cache] failed to write {path}: {e}")
