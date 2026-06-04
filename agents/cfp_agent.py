import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import time
import random
import ssl
import urllib3
from datetime import date

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.state import PipelineState
from core.registry import get_conference, update_cfp_url
from core.cache import hash_content, cache_get, cache_set

from config import GROQ_MODEL

from json_repair import repair_json

HEAD_CHARS = 3000
TAIL_CHARS = 3000
KEYWORD_CTX = 400
TOPIC_KEYWORD_CTX = 3500
DEADLINE_KEYWORDS = [
    "deadline", "due date", "submission deadline",
    "paper deadline", "abstract deadline", "full paper due",
]
TOPIC_KEYWORDS = [
    "topics of interest",
    "areas of interest",
    "topic areas",
    "call for papers",
    "we solicit",
    "we invite submissions",
    "scope of the conference",
    "topics include",
    "research areas",
]

HEADERS = {
    "User-Agent": "CFPScraper/1.0 (research@example.com)",
    "Accept-Encoding": "gzip, deflate",
}


# HTTP helper

def _safe_get_persistent(url: str, max_attempts: int = 3) -> requests.Response | None:
    ctx = ssl.create_default_context()
    ctx.set_ciphers("DEFAULT")
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter()
    session.mount("https://", adapter)

    for attempt in range(max_attempts):
        try:
            r = session.get(url, headers=HEADERS, timeout=20, verify=False)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  [cfp_agent] 429 on {url} — sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue
            r.raise_for_status()
        except Exception as e:
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"  [cfp_agent] fetch error ({url}): {e} — retrying in {sleep_time:.1f}s")
            time.sleep(sleep_time)
    return None


def _safe_get(url: str, max_attempts: int = 3) -> requests.Response | None:
    """
    GET with retries. Skips retry on 4xx errors (except 429), since those
    are not transient — retrying a 404 just wastes ~7s.
    """
    ctx = ssl.create_default_context()
    ctx.set_ciphers("DEFAULT")
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter()
    session.mount("https://", adapter)

    for attempt in range(max_attempts):
        try:
            r = session.get(url, headers=HEADERS, timeout=20, verify=False)

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  [cfp_agent] 429 on {url} — sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue

            # Don't retry permanent 4xx errors
            if 400 <= r.status_code < 500 and r.status_code != 429:
                print(f"  [cfp_agent] {r.status_code} on {url} — not retrying (client error)")
                return None

            # 5xx or other — let it raise and be caught below for backoff
            r.raise_for_status()

        except Exception as e:
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"  [cfp_agent] fetch error ({url}): {e} — retrying in {sleep_time:.1f}s")
            time.sleep(sleep_time)
    return None

# text extraction

def _extract_text(html: str) -> str:
    """
    Head + keyword-context (deadlines + topics) + tail.
    Deadlines need short context; topic lists need much longer context.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    parts = [text[:HEAD_CHARS]]
    lower = text.lower()
    seen_ranges: list[tuple[int, int]] = []

    def _add_keyword_context(keywords: list[str], ctx: int, label: str):
        for kw in keywords:
            idx = 0
            while True:
                pos = lower.find(kw, idx)
                if pos == -1:
                    break
                start = max(0, pos - 100)
                end = min(len(text), pos + ctx)
                # Skip if already covered by a previously added range
                if not any(s <= start and end <= e for s, e in seen_ranges):
                    parts.append(f"[...'{kw}' ({label}) context...]\n" + text[start:end])
                    seen_ranges.append((start, end))
                idx = pos + 1

    # topics first, take priority if ranges overlap
    _add_keyword_context(TOPIC_KEYWORDS, TOPIC_KEYWORD_CTX, "topic")
    _add_keyword_context(DEADLINE_KEYWORDS, KEYWORD_CTX, "deadline")

    if len(text) > HEAD_CHARS:
        parts.append("[...page end...]\n" + text[-TAIL_CHARS:])

    return "\n\n".join(parts)


# LLM setup

def _build_llm() -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=1024,
    )


# prompts

_HOMEPAGE_PROMPT = """\
You are a structured data extractor reading the homepage of the {conf_name} conference website.

Your only job here is to find submission deadlines. Some conferences have TWO submission
cycles (Round 1 and Round 2) — extract ALL of them.

Return ONLY a valid JSON object, no preamble, no markdown fences:
{{
  "deadlines": [
    {{"cycle": 1, "label": "Full Paper Submission", "date": "YYYY-MM-DD"}},
    {{"cycle": 2, "label": "Round 2 Submission",    "date": "YYYY-MM-DD"}}
  ]
}}

Rules:
- Normalize all dates to YYYY-MM-DD. If no year is shown, assume {year}.
- Prefer full paper deadlines over abstract-only deadlines.
- If only one deadline exists, return a single-element list.
- Return an empty list if no deadlines are found at all.

Page text:
{text}
"""

_CFP_PROMPT = """\
You are a structured data extractor reading the Call for Papers (CFP) page of {conf_name}.

Extract:
1. "topics" — up to 20 short topic phrases listed in the CFP. Return [] if not found.
2. "deadlines" — any submission deadlines on this page (same format as homepage extraction).
   Return [] if none found here.
3. "cfp_url_hint" — if this page clearly links to a more specific or complete CFP page,
   return that URL as a string. Otherwise return null.

Return ONLY a valid JSON object, no preamble, no markdown fences:
{{
  "topics": ["topic 1", "topic 2"],
  "deadlines": [
    {{"cycle": 1, "label": "...", "date": "YYYY-MM-DD"}}
  ],
  "cfp_url_hint": null
}}

Page text:
{text}
"""

# LLM extraction

def _parse_llm(raw: str) -> dict:
    return json.loads(repair_json(raw))

def _llm_extract(
    llm: ChatGroq,
    text: str,
    prompt_template: str,
    conf_name: str,
    year: int,
    page_kind: str = "page",
    url: str | None = None,
) -> dict:
    """
    Two-tier cache:
    - Content-hash cache: catches pages with same content under different URLs
    - URL cache (written here, read upstream in _scrape_one before fetching)
    """
    cache_key = f"{conf_name}_{year}_{page_kind}_{hash_content(text)}"
    cached = cache_get("cfp_extract", cache_key)
    if cached is not None:
        print(f"  [cfp_agent] {conf_name}: cache hit ({page_kind}, content) → skipping LLM")
        return cached

    prompt = prompt_template.format(
        conf_name=conf_name.upper(),
        year=year,
        text=text,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = _parse_llm(response.content)
    except json.JSONDecodeError as e:
        print(f"  [cfp_agent] JSON parse error for {conf_name}: {e}")
        return {}
    except Exception as e:
        print(f"  [cfp_agent] LLM error for {conf_name}: {e}")
        return {}

    # Only cache if extraction yielded something useful
    has_content = result and (
        result.get("deadlines") or
        result.get("topics") or
        result.get("cfp_url_hint")
    )
    if has_content:
        cache_set("cfp_extract", cache_key, result)
        if url:
            _cache_set_by_url(conf_name, year, page_kind, url, result)
        print(f"  [cfp_agent] {conf_name}: cached {page_kind} extraction")

    return result


# helpers

def _substitute_year(url: str, year: int) -> str:
    """Replace any 4-digit year in the URL with target year."""
    return re.sub(r"\b20\d{2}\b", str(year), url)


def _merge_deadlines(a: list[dict], b: list[dict]) -> list[dict]:
    """Union of two deadline lists, deduplicated by date, sorted ascending."""
    seen = set()
    merged = []
    for d in a + b:
        dt = d.get("date")
        if dt and dt not in seen:
            seen.add(dt)
            merged.append(d)
    return sorted(merged, key=lambda x: x.get("date") or "")

def _cache_get_by_url(conf_name: str, year: int, page_kind: str, url: str) -> dict | None:
    """Look up cached result by URL alone (no fetch needed)."""
    key = f"{conf_name}_{year}_{page_kind}_url_{hash_content(url)}"
    return cache_get("cfp_extract", key)


def _cache_set_by_url(conf_name: str, year: int, page_kind: str, url: str, result: dict) -> None:
    """Store result under URL key (so next run can skip the fetch)."""
    key = f"{conf_name}_{year}_{page_kind}_url_{hash_content(url)}"
    cache_set("cfp_extract", key, result)


# per-conference scrape

def _scrape_one(conf_name: str, conf: dict, llm: ChatGroq, cfp_year: int) -> dict:
    """
    Fetch homepage + CFP page for a single conference and extract
    deadlines (merged from both) and topics (from CFP page).
    Returns: {deadlines, topics, url, error}
    """
    # manual overrides first (for SPA conference pages)
    override_deadlines = conf.get("cfp_deadlines_override") or []
    override_topics    = conf.get("cfp_topics_override") or []

    if override_deadlines or override_topics:
        print(
            f"  [cfp_agent] {conf_name}: using manual override "
            f"({len(override_deadlines)} deadline(s), {len(override_topics)} topic(s))"
        )
        return {
            "deadlines": override_deadlines,
            "topics":    override_topics,
            "url":       conf.get("cfp_url") or conf.get("url"),
            "error":     None,
        }

    homepage_url = conf.get("url")
    cfp_url = conf.get("cfp_url")
    deadlines: list[dict] = []
    topics: list[str] = []
    used_url = cfp_url or homepage_url

    # 1. homepage — deadline extraction
    if homepage_url:
        url_with_year = _substitute_year(homepage_url, cfp_year)

        cached = _cache_get_by_url(conf_name, cfp_year, "homepage", url_with_year)
        if cached is not None:
            print(f"  [cfp_agent] {conf_name}: cache hit (homepage, url) → skipping fetch")
            deadlines = cached.get("deadlines", [])
        else:
            print(f"  [cfp_agent] {conf_name}: homepage → {url_with_year}")
            r = _safe_get(url_with_year)
            if r:
                result = _llm_extract(
                    llm, _extract_text(r.text), _HOMEPAGE_PROMPT,
                    conf_name, cfp_year,
                    page_kind="homepage", url=url_with_year,
                )
                deadlines = result.get("deadlines", [])
            else:
                print(f"  [cfp_agent] {conf_name}: homepage fetch failed")

        print(f"  [cfp_agent] {conf_name}: homepage → {len(deadlines)} deadline(s)")

    # 2. CFP page — topics + supplementary deadlines
    if cfp_url:
        cfp_with_year = _substitute_year(cfp_url, cfp_year)

        cached = _cache_get_by_url(conf_name, cfp_year, "cfp", cfp_with_year)
        if cached is not None:
            print(f"  [cfp_agent] {conf_name}: cache hit (cfp, url) → skipping fetch")
            topics = cached.get("topics", [])
            deadlines = _merge_deadlines(deadlines, cached.get("deadlines", []))
            used_url = cfp_with_year
            #  hint URL that was followed at original-cache time
            hint_from_cache = cached.get("cfp_url_hint")
            if hint_from_cache:
                hint_cached = _cache_get_by_url(conf_name, cfp_year, "cfp_hint", hint_from_cache)
                if hint_cached:
                    print(f"  [cfp_agent] {conf_name}: cache hit (cfp_hint, url) → skipping fetch")
                    if hint_cached.get("topics"):
                        topics = hint_cached["topics"]
                    deadlines = _merge_deadlines(deadlines, hint_cached.get("deadlines", []))
                    used_url = hint_from_cache
        else:
            print(f"  [cfp_agent] {conf_name}: CFP page → {cfp_with_year}")
            r2 = _safe_get(cfp_with_year)
            if r2:
                result2 = _llm_extract(
                    llm, _extract_text(r2.text), _CFP_PROMPT,
                    conf_name, cfp_year,
                    page_kind="cfp", url=cfp_with_year,
                )
                topics = result2.get("topics", [])
                deadlines = _merge_deadlines(deadlines, result2.get("deadlines", []))
                used_url = cfp_with_year
                print(
                    f"  [cfp_agent] {conf_name}: CFP page → "
                    f"{len(topics)} topic(s), "
                    f"{len(result2.get('deadlines', []))} extra deadline(s)"
                )

                # follow the hint if LLM found a more specific CFP page
                hint = result2.get("cfp_url_hint")
                if hint and hint.rstrip("/") != cfp_with_year.rstrip("/"):
                    cached_hint = _cache_get_by_url(conf_name, cfp_year, "cfp_hint", hint)
                    if cached_hint is not None:
                        print(f"  [cfp_agent] {conf_name}: cache hit (cfp_hint, url) → skipping fetch")
                        if cached_hint.get("topics"):
                            topics = cached_hint["topics"]
                        deadlines = _merge_deadlines(deadlines, cached_hint.get("deadlines", []))
                        used_url = hint
                    else:
                        print(f"  [cfp_agent] {conf_name}: following hint → {hint}")
                        r3 = _safe_get(hint)
                        if r3:
                            result3 = _llm_extract(
                                llm, _extract_text(r3.text), _CFP_PROMPT,
                                conf_name, cfp_year,
                                page_kind="cfp_hint", url=hint,
                            )
                            if result3.get("topics"):
                                topics = result3["topics"]
                            deadlines = _merge_deadlines(deadlines, result3.get("deadlines", []))
                            used_url = hint
                            try:
                                update_cfp_url(conf_name, hint)
                                print(f"  [cfp_agent] {conf_name}: persisted hint URL to registry")
                            except Exception as e:
                                print(f"  [cfp_agent] {conf_name}: could not persist hint URL: {e}")
            else:
                print(f"  [cfp_agent] {conf_name}: CFP page fetch failed")

    error = None if (deadlines or topics) else "no data extracted"
    return {"deadlines": deadlines, "topics": topics, "url": used_url, "error": error}


#  LangGraph node

def cfp_scraper_node(state: PipelineState) -> dict:
    """
    LangGraph node: scrape CFP pages and extract deadlines + topics.

    Reads:  conferences_in_scope, conference_metadata
    Writes: cfp_data, errors
    """
    conferences: list[str] = state["conferences_in_scope"]
    conference_metadata: dict = state["conference_metadata"]

    # target the upcoming conference year; fall back to current year per conference
    primary_year = date.today().year + 1
    fallback_year = date.today().year

    print(f"\n[cfp_agent] target year: {primary_year} (fallback: {fallback_year})")
    print(f"[cfp_agent] conferences: {conferences}")

    llm = _build_llm()
    cfp_data: dict = {}
    errors: list = []

    for conf_name in conferences:
        # prefer metadata already in state; re-query registry as safety net
        conf = conference_metadata.get(conf_name) or get_conference(conf_name)
        if not conf:
            msg = f"cfp_agent [{conf_name}]: not found in registry"
            errors.append(msg)
            print(f"  WARNING {msg}")
            cfp_data[conf_name] = {"deadlines": [], "topics": [], "url": None}
            continue

        result = _scrape_one(conf_name, conf, llm, primary_year)

        # graceful fallback to current year if nothing came back
        if result["error"]:
            print(f"  [cfp_agent] {conf_name}: no data for {primary_year}, retrying {fallback_year}")
            result = _scrape_one(conf_name, conf, llm, fallback_year)

        if result["error"]:
            msg = f"cfp_agent [{conf_name}]: {result['error']}"
            errors.append(msg)
            print(f"  WARNING {msg}")

        cfp_data[conf_name] = {
            "deadlines": result["deadlines"],
            "topics":    result["topics"],
            "url":       result["url"],
        }

        print(
            f"  [cfp_agent] {conf_name}: "
            f"{len(result['deadlines'])} deadline(s), "
            f"{len(result['topics'])} topic(s)"
        )

        time.sleep(random.uniform(2, 4))

    return {
        "cfp_data": cfp_data,
        "errors":   errors,
    }
