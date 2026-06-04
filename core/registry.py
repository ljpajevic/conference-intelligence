# core/registry.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import DB_PATH


# schema

DDL = """
CREATE TABLE IF NOT EXISTS conferences (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT UNIQUE NOT NULL,
    full_name               TEXT NOT NULL,
    url                     TEXT,
    cfp_url                 TEXT,
    dblp_key                TEXT,
    venue_type              TEXT,
    topic_areas_seed        TEXT DEFAULT '[]',
    topic_areas_discovered  TEXT DEFAULT '[]',
    typical_months          TEXT DEFAULT '[]',
    cfp_deadlines_override  TEXT DEFAULT '[]',
    cfp_topics_override     TEXT DEFAULT '[]',
    last_scraped            TIMESTAMP,
    added_by                TEXT DEFAULT 'seed'
);
"""

SEED_DATA = [
    {
        "name": "sigcomm",
        "full_name": "ACM SIGCOMM",
        "url": "https://conferences.sigcomm.org/sigcomm/2026/",
        "cfp_url": "https://conferences.sigcomm.org/sigcomm/2026/cfp/",
        "dblp_key": "sigcomm",
        "venue_type": "networking",
        "topic_areas_seed": ["networking", "protocols", "datacenter", "transport", "SDN"],
        "typical_months": [1, 2],
    },
    {
        "name": "imc",
        "full_name": "ACM Internet Measurement Conference",
        "url": "https://conferences.sigcomm.org/imc/2026/",
        "cfp_url": "https://conferences.sigcomm.org/imc/2026/cfp",
        "dblp_key": "imc",
        "venue_type": "networking",
        "topic_areas_seed": ["measurement", "traffic analysis", "internet topology", "security"],
        "typical_months": [3, 4],
    },
    {
        "name": "conext",
        "full_name": "ACM CoNEXT",
        "url": "https://conferences.sigcomm.org/co-next/2026",
        "cfp_url": "https://conferences.sigcomm.org/co-next/2026/#!/cfp",
        "dblp_key": "conext",
        "venue_type": "networking",
        "topic_areas_seed": ["networking", "protocols", "wireless", "CDN"],
        "typical_months": [6, 7],
        "dblp_source": "pacmnet",
    },
    {
        "name": "infocom",
        "full_name": "IEEE INFOCOM",
        "url": "https://infocom2025.ieee-infocom.org",
        "cfp_url": "https://infocom2025.ieee-infocom.org/authors/call-papers-main-conference",
        "dblp_key": "infocom",
        "venue_type": "networking",
        "topic_areas_seed": ["networking", "wireless", "IoT", "QoS", "5G"],
        "typical_months": [7, 8],
    },
    {
        "name": "mobicom",
        "full_name": "ACM MobiCom",
        "url": "https://sigmobile.org/mobicom/2026/index.html",
        "cfp_url": "https://sigmobile.org/mobicom/2026/cfp.html",
        "dblp_key": "mobicom",
        "venue_type": "mobile",
        "topic_areas_seed": ["mobile", "wireless", "sensing", "5G", "mmWave"],
        "typical_months": [3, 4],
    },
    {
        "name": "mobisys",
        "full_name": "ACM MobiSys",
        "url": "https://www.sigmobile.org/mobisys/2026/",
        "cfp_url": "https://www.sigmobile.org/mobisys/2026/call_for_papers/",
        "dblp_key": "mobisys",
        "venue_type": "mobile",
        "topic_areas_seed": ["mobile systems", "wearables", "sensing", "edge", "AR/VR"],
        "typical_months": [1, 2],
    },
    {
        "name": "eurosys",
        "full_name": "EuroSys",
        "url": "https://2026.eurosys.org/",
        "cfp_url": "https://2026.eurosys.org/cfp.html#calls",
        "dblp_key": "eurosys",
        "venue_type": "systems",
        "topic_areas_seed": ["systems", "OS", "distributed systems", "storage", "virtualization"],
        "typical_months": [10, 11],
    },
    {
        "name": "icdcs",
        "full_name": "IEEE ICDCS",
        "url": "https://icdcs2026.icdcs.org/",
        "cfp_url": "https://icdcs2026.icdcs.org/calls/call-for-papers/",
        "dblp_key": "icdcs",
        "venue_type": "systems",
        "topic_areas_seed": ["distributed systems", "cloud", "edge", "consensus", "fault tolerance"],
        "typical_months": [1, 2],
    },
]


# database connection

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db() -> None:
    """Create tables and seed data if not already present."""
    with get_connection() as conn:
        conn.execute(DDL)
        conn.commit()
        _seed(conn)
    print(f"Registry initialized at {DB_PATH}")


def _seed(conn: sqlite3.Connection) -> None:
    for conf in SEED_DATA:
        existing = conn.execute(
            "SELECT id FROM conferences WHERE name = ?",
            (conf["name"],)
        ).fetchone()

        if existing:
            continue  # never overwrite existing records!

        conn.execute("""
            INSERT INTO conferences (
                name, full_name, url, cfp_url, dblp_key,
                venue_type, topic_areas_seed, typical_months, added_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'seed')
        """, (
            conf["name"],
            conf["full_name"],
            conf["url"],
            conf["cfp_url"],
            conf["dblp_key"],
            conf["venue_type"],
            json.dumps(conf["topic_areas_seed"]),
            json.dumps(conf["typical_months"]),
            conf.get("dblp_source", "conf"),
        ))

    conn.commit()
    print(f"Seeded {len(SEED_DATA)} conferences")


# query functions

def list_conferences() -> list[dict]:
    """
    Return all conferences as full row dicts (including dblp_key, urls, etc.).
    Use this when callers need more than basic metadata.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM conferences ORDER BY name
        """).fetchall()
    return [_deserialize(dict(r)) for r in rows]


def get_conference(name: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM conferences WHERE name = ?",
            (name.lower(),)
        ).fetchone()
    return _deserialize(dict(row)) if row else None


def add_conference(
    name: str,
    full_name: str,
    dblp_key: str,
    venue_type: str,
    url: Optional[str] = None,
    cfp_url: Optional[str] = None,
    topic_areas_seed: Optional[list] = None,
    typical_months: Optional[list] = None,
) -> dict:
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conferences (
                name, full_name, url, cfp_url, dblp_key,
                venue_type, topic_areas_seed, typical_months, added_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'user')
        """, (
            name.lower(),
            full_name,
            url,
            cfp_url,
            dblp_key,
            venue_type,
            json.dumps(topic_areas_seed or []),
            json.dumps(typical_months or []),
        ))
        conn.commit()
    return get_conference(name)

def delete_conference(name: str) -> bool:
    """Remove a conference from the registry. Returns True if a row was deleted."""
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM conferences WHERE name = ?",
            (name.lower(),)
        )
        conn.commit()
    return cur.rowcount > 0

def update_discovered_topics(name: str, topics: list[str]) -> dict:
    with get_connection() as conn:
        conn.execute("""
            UPDATE conferences
            SET topic_areas_discovered = ?,
                last_scraped = ?
            WHERE name = ?
        """, (
            json.dumps(topics),
            datetime.utcnow().isoformat(),
            name.lower(),
        ))
        conn.commit()
    return get_conference(name)


def update_cfp_url(name: str, cfp_url: str) -> dict:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conferences SET cfp_url = ? WHERE name = ?",
            (cfp_url, name.lower())
        )
        conn.commit()
    return get_conference(name)


#  helpers

def _deserialize(row: dict) -> dict:
    """Parse JSON columns back to Python lists."""
    for col in (
        "topic_areas_seed",
        "topic_areas_discovered",
        "typical_months",
        "cfp_deadlines_override",
        "cfp_topics_override",
    ):
        if col in row and isinstance(row[col], str):
            row[col] = json.loads(row[col])
    return row

def update_cfp_overrides(
    name: str,
    deadlines: list[dict] | None = None,
    topics: list[str] | None = None,
) -> dict:
    """
    Manually set CFP deadlines and/or topics for a conference.
    Used for JS-rendered SPA conferences where scraping returns nothing.
    Pass None to leave a field unchanged; pass [] to clear it.
    """
    updates = []
    params = []
    if deadlines is not None:
        updates.append("cfp_deadlines_override = ?")
        params.append(json.dumps(deadlines))
    if topics is not None:
        updates.append("cfp_topics_override = ?")
        params.append(json.dumps(topics))
    if not updates:
        return get_conference(name)

    params.append(name.lower())
    with get_connection() as conn:
        conn.execute(
            f"UPDATE conferences SET {', '.join(updates)} WHERE name = ?",
            params,
        )
        conn.commit()
    return get_conference(name)
