"""
Schema migration script to add dblp_source column to the conferences table.

"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from config import DB_PATH

def main():
    with sqlite3.connect(DB_PATH) as conn:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(conferences)")}
        if "dblp_source" not in existing_cols:
            conn.execute(
                "ALTER TABLE conferences ADD COLUMN dblp_source TEXT DEFAULT 'conf'"
            )
            print("Added column: dblp_source")
        else:
            print("Column dblp_source already exists, skipping")
        conn.commit()
    print("Migration complete.")

if __name__ == "__main__":
    main()
