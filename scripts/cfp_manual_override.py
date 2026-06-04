"""
Manual CFP override script.

Use when automatic CFP scraping fails for a conference.
Common causes:
  - CFP page uses JavaScript rendering (not handled by static fetching)
  - Page structure changed and LLM extraction returns partial data
  - Conference homepage doesn't expose deadlines until close to submission

Workflow:
  1. Manually copy deadlines and topics from the official CFP page
  2. Update the constants below
  3. Run: python scripts/cfp_manual_override.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import update_cfp_overrides


# which conference to override
CONFERENCE = "conext"

# deadlines
# some confs list multiple submission cycles, list each separately
# date format: YYYY-MM-DD
DEADLINES = [
    {"cycle": 1, "label": "Paper Registration", "date": "2025-12-05"},
    {"cycle": 1, "label": "Paper Submission",   "date": "2025-12-12"},
    {"cycle": 2, "label": "Paper Registration", "date": "2026-05-29"},
    {"cycle": 2, "label": "Paper Submission",   "date": "2026-06-06"},
]

# topics extracted from CFP
# copy verbatim from the conference's "Topics of Interest" section
# because this gives the strongest signal for matching
TOPICS = [
    "networked systems",
    "internet architecture",
    "wireless and mobile networks",
    "data center networks",
    "network security",
]


def main():
    update_cfp_overrides(
        CONFERENCE,
        deadlines=DEADLINES,
        topics=TOPICS,
    )
    print(f"✓ {CONFERENCE.upper()} CFP overrides applied.")


if __name__ == "__main__":
    main()
