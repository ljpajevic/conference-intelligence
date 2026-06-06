# ============================================================
# Networking Conference Paper Pipeline
#
# Conferences:
#   - SIGCOMM
#   - CoNEXT
#   - IMC
#   - Infocom
#   - MobiCom
#   - MobiSys
#   - EuroSys
#   - ICDCS
#
# Features:
#   - DBLP XML scraping
#   - Async OpenAlex enrichment (aiohttp + semaphore)
#   - DOI-keyed shelve cache (reruns are instant)
#   - Abstract retrieval
#   - Citation counts
#   - PDF URLs
#   - Semantic chunking
#   - Parquet export
#
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import random
import re
import shelve
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from core.tools import compute_embeddings, build_paper_text
from core.registry import list_conferences

# ============================================================
# CONFIG
# ============================================================

N_YEARS = 3

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
ENRICHED_DIR = DATA_DIR / "enriched"
CHUNKS_DIR = DATA_DIR / "chunks"
CACHE_PATH = str(DATA_DIR / "openalex_cache")

RAW_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "PaperPipeline/1.0 (research@example.com)",
    "Accept-Encoding": "gzip, deflate",  # explicitly exclude br
}

NON_PAPER_PREFIXES = (
    "poster:", "demo:", "demonstration:", "wip:", "work-in-progress:",
    "tutorial:", "panel:", "workshop:", "keynote:", "abstract:",
    "late breaking", "late-breaking",
)

MIN_PAGES = 6

# limit the number of parallel requests to OpenAlex
# 20 is well within their polite use ceiling
OPENALEX_MAX_CONCURRENT = 20


def get_conferences() -> dict:
    """Build conference name → DBLP key map from the registry."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.registry import list_conferences
    return {c["name"]: c["dblp_key"] for c in list_conferences()}


# DBLP XML fetching

def dblp_xml_url(conf_key: str, year: int) -> str:
    return f"https://dblp.org/db/conf/{conf_key}/{conf_key}{year}.xml"


def safe_get(url, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"429 hit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue
            r.raise_for_status()
        except ConnectionResetError:
            sleep_time = (2 ** attempt) * 5  # ← longer backoff for resets
            print(f"Connection reset, sleeping {sleep_time}s before retry")
            time.sleep(sleep_time)
        except Exception as e:
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"ERROR: {e}")
            time.sleep(sleep_time)
    return None


def get_last_n_valid_years(conf_key: str, n: int = 5) -> list[int]:
    """Scrape DBLP index page to find years that actually have proceedings."""
    index_url = f"https://dblp.org/db/conf/{conf_key}/index.html"
    r = safe_get(index_url)
    if r is None:
        print(f"Could not fetch index for {conf_key}")
        return []
    years = re.findall(rf"{conf_key}(\d{{4}})\.html", r.text)
    years = sorted(set(int(y) for y in years))
    return years[-n:]


def parse_dblp_xml(conf_name: str, conf_key: str, year: int) -> list[dict]:
    """Fetch and parse one conference-year's DBLP XML. Returns list of paper dicts."""
    url = dblp_xml_url(conf_key, year)
    print(f"Fetching {url}")

    response = safe_get(url)

    # None check has to be before accessing .status_code
    if response is None:
        print(f"FAILED: {url}")
        return []

    if response.status_code != 200 or len(response.content) < 1000:
        print(f"Skipping empty/invalid response: {url}")
        return []

    try:
        root = ET.fromstring(response.content)
    except Exception as e:
        print(f"XML parse error for {url}: {e}")
        return []

    papers = []
    for item in root.findall(".//inproceedings"):
        if not _is_full_paper(item):
            continue
        title = item.findtext("title")
        year_text = item.findtext("year")
        authors = [a.text for a in item.findall("author") if a.text]

        doi = None
        ee = item.findtext("ee")
        if ee and "doi.org" in ee:
            doi = ee

        papers.append({
            "conference": conf_name,
            "conference_key": conf_key,
            "title": title,
            "year": year_text,
            "authors": authors,
            "doi": doi,
            "dblp_url": ee,
        })

    print(f"  → {conf_name} {year}: {len(papers)} papers")
    return papers

def parse_pacmnet_xml(conf_name: str, year: int, issue_prefix: str = "CoNEXT") -> list[dict]:
    """
    Fetch and parse a PACMNET volume XML for a specific conference's papers.

    PACMNET is a journal that hosts full papers for several ACM venues
    (CoNEXT, etc.), distinguished by the `<number>` field (e.g., "CoNEXT1").
    Volumes correspond to years: volume 1 = 2023, volume 2 = 2024, ...
    """
    volume = year - 2022   # volume 1 starts in 2023
    url = f"https://dblp.org/db/journals/pacmnet/pacmnet{volume}.xml"
    print(f"Fetching {url}")

    response = safe_get(url)
    if response is None:
        print(f"FAILED: {url}")
        return []

    if response.status_code != 200 or len(response.content) < 1000:
        print(f"Skipping empty/invalid response: {url}")
        return []

    try:
        root = ET.fromstring(response.content)
    except Exception as e:
        print(f"XML parse error for {url}: {e}")
        return []

    papers = []
    for art in root.findall(".//article"):
        number = (art.findtext("number") or "").strip()
        if not number.startswith(issue_prefix):
            continue   # skip articles not from this conference's track

        title = art.findtext("title") or ""
        # skip keynotes, editorials, and such
        if title.lower().startswith(("pacmnet", "editorial")):
            continue

        year_text = art.findtext("year")
        authors = [a.text for a in art.findall("author") if a.text]

        doi = None
        ee = art.findtext("ee")
        if ee and "doi.org" in ee:
            doi = ee

        papers.append({
            "conference": conf_name,
            "conference_key": f"pacmnet/{number}",
            "title": title,
            "year": year_text,
            "authors": authors,
            "doi": doi,
            "dblp_url": ee,
        })

    print(f"  → {conf_name} {year} (via PACMNET v{volume}): {len(papers)} full papers")
    return papers

def _is_full_paper(item) -> bool:
    """
    Filter out posters, demos, short papers, and other non-research entries
    that DBLP groups under <inproceedings> alongside main-track papers.
    """
    title = (item.findtext("title") or "").strip().lower()
    if any(title.startswith(p) for p in NON_PAPER_PREFIXES):
        return False

    pages = item.findtext("pages") or ""
    if pages:
        if "-" in pages:
            try:
                start, end = pages.split("-")
                if int(end) - int(start) + 1 < MIN_PAGES:
                    return False
            except ValueError:
                pass
        else:
            # single page like "1" or "2"; likely just a keynote title
            return False

    return True

def collect_all_papers() -> pd.DataFrame:
    from core.registry import list_conferences
    all_papers = []

    for conf in list_conferences():
        conf_name = conf["name"]
        conf_key = conf["dblp_key"]
        source = conf.get("dblp_source", "conf")

        if source == "pacmnet":
            # PACMNET volumes correspond to years (v1 = 2023)
            # don't need to probe the DBLP index; use the last N_YEARS
            from datetime import date
            this_year = date.today().year
            years = list(range(this_year - N_YEARS, this_year + 1))
            print(f"\n{conf_name} (PACMNET) years to fetch: {years}")
            for year in years:
                papers = parse_pacmnet_xml(conf_name, year)
                all_papers.extend(papers)
                time.sleep(random.uniform(5, 10))
        else:
            valid_years = get_last_n_valid_years(conf_key, N_YEARS)
            print(f"\n{conf_name} valid years: {valid_years}")
            for year in valid_years:
                papers = parse_dblp_xml(conf_name, conf_key, year)
                all_papers.extend(papers)
                time.sleep(random.uniform(5, 10))

        print(f"Cooling down after {conf_name}...")
        time.sleep(random.uniform(15, 25))

    return pd.DataFrame(all_papers)


# OpenAlex enrichment - async

def normalize_doi(doi: str) -> str:
    return (
        doi
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
    )


def openalex_url_from_doi(doi: str) -> str:
    clean = normalize_doi(doi)
    return f"https://api.openalex.org/works/https://doi.org/{clean}"


def reconstruct_abstract(inv_idx: dict | None) -> str | None:
    if not inv_idx:
        return None
    words = []
    for word, positions in inv_idx.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join(word for _, word in words)


def parse_openalex_work(work: dict) -> dict:
    """Extract the fields we care about from an OpenAlex work object."""
    return {
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "citation_count": work.get("cited_by_count"),
        "concepts": [c["display_name"] for c in work.get("concepts", [])],
        "pdf_url": work.get("primary_location", {}).get("pdf_url"),
        "openalex_id": work.get("id"),
    }


def empty_enrichment() -> dict:
    return {
        "abstract": None,
        "citation_count": None,
        "concepts": None,
        "pdf_url": None,
        "openalex_id": None,
    }


async def fetch_openalex_async(
    session: aiohttp.ClientSession,
    doi: str,
    semaphore: asyncio.Semaphore,
    cache: shelve.Shelf,
) -> dict:
    """
    Fetch one OpenAlex record.
    Checks the shelve cache first; stores result on success.
    Returns a parsed enrichment dict (never None).
    """
    if not doi:
        return empty_enrichment()

    cache_key = normalize_doi(doi)

    # cache hit, no network call needed
    if cache_key in cache:
        return cache[cache_key]

    url = openalex_url_from_doi(doi)

    async with semaphore:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    return empty_enrichment()
                work = await r.json(content_type=None)
                result = parse_openalex_work(work)
                cache[cache_key] = result  # persist for future runs
                return result
        except Exception as e:
            print(f"OpenAlex fetch error for {doi}: {e}")
            return empty_enrichment()


async def enrich_papers_async(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich all papers in parallel (up to OPENALEX_MAX_CONCURRENT at once).
    Uses a shelve cache so reruns skip already-fetched DOIs.
    """
    semaphore = asyncio.Semaphore(OPENALEX_MAX_CONCURRENT)

    with shelve.open(CACHE_PATH) as cache:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            tasks = [
                fetch_openalex_async(session, row["doi"], semaphore, cache)
                for _, row in df.iterrows()
            ]

            print(f"Fetching {len(tasks)} OpenAlex records "
                  f"({OPENALEX_MAX_CONCURRENT} concurrent)...")

            results = await async_tqdm.gather(*tasks)

    enriched_rows = []
    for (_, row), enrichment in zip(df.iterrows(), results):
        enriched_rows.append({**row.to_dict(), **enrichment})

    return pd.DataFrame(enriched_rows)


# document building

def build_document(row: pd.Series) -> str:
    authors = ", ".join(row["authors"] if isinstance(row["authors"], list) else json.loads(row["authors"] or "[]"))
    concepts = ", ".join(row["concepts"] if isinstance(row["concepts"], list) else json.loads(row["concepts"] or "[]"))
    return f"""TITLE:
{row['title']}

AUTHORS:
{authors}

VENUE:
{row['conference'].upper()} {row['year']}

ABSTRACT:
{row['abstract'] or 'N/A'}

CONCEPTS:
{concepts}
"""


# chunking

def create_chunks(df: pd.DataFrame) -> pd.DataFrame:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        doc = build_document(row)
        split_texts = splitter.split_text(doc)
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "chunk_id": f"{row['conference']}_{row['year']}_{row.name}_{i}",
                "paper_title": row["title"],
                "conference": row["conference"],
                "year": row["year"],
                "doi": row["doi"],
                "text": chunk,
            })

    return pd.DataFrame(chunks)



# helpers

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize list columns to JSON strings for Parquet compatibility."""
    df = df.copy()
    for col in ["authors", "concepts"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    normalize_columns(df).to_parquet(path, index=False)
    print(f"Saved {len(df)} rows → {path}")


# calculate coverage

def print_coverage_report(df: pd.DataFrame) -> None:
    """Show abstract coverage per conference per year."""
    print("\n=== Abstract Coverage Report ===")
    total = len(df)
    has_abstract = df["abstract"].notna().sum()
    print(f"Overall: {has_abstract}/{total} ({has_abstract/total:.1%})\n")

    for (conf, year), group in df.groupby(["conference", "year"]):
        n = len(group)
        n_abs = group["abstract"].notna().sum()
        bar = "█" * int(n_abs / n * 20) + "░" * (20 - int(n_abs / n * 20))
        print(f"  {conf:10} {year}  [{bar}]  {n_abs}/{n} ({n_abs/n:.0%})")


def validate_conference_keys(conferences):
    """Test each conference's DBLP URL. Works for both conf and PACMNET sources."""
    for conf in conferences:
        name = conf["name"]
        source = conf.get("dblp_source", "conf")

        if source == "pacmnet":
            url = "https://dblp.org/db/journals/pacmnet/index.html"
        else:
            url = f"https://dblp.org/db/conf/{conf['dblp_key']}/index.html"

        r = safe_get(url)
        status = "✓" if r and r.status_code == 200 else "✗ INVALID"
        print(f"{name:12} ({source:7}) {status}")
        time.sleep(random.uniform(2, 3))

# embeddings

def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute embeddings for title + abstract and add them as a column.
    Embedding stored as a list per row for parquet compatibility.
    """
    df = df.copy()
    texts = [
        build_paper_text(row.get("title"), row.get("abstract"))
        for _, row in df.iterrows()
    ]
    print(f"Computing embeddings for {len(texts)} papers…")
    embeddings = compute_embeddings(texts)
    df["embedding"] = list(embeddings)
    return df

# -- main

def main() -> None:

    from core.registry import list_conferences
    validate_conference_keys(list_conferences())

    print("\n--------------------")
    print("STEP 1: Collect DBLP papers")
    print("--------------------\n")

    raw_df = collect_all_papers()

    if raw_df.empty:
        print("No papers collected — check conference keys and network.")
        return

    save_dataframe(raw_df, RAW_DIR / "networking_papers_raw.parquet")
    print(f"\nTotal papers collected: {len(raw_df)}")

    print("\n--------------------")
    print("STEP 2: OpenAlex enrichment")
    print("--------------------\n")

    enriched_df = asyncio.run(enrich_papers_async(raw_df))

    save_dataframe(enriched_df, ENRICHED_DIR / "networking_papers_enriched.parquet")

    print_coverage_report(enriched_df)

    save_dataframe(enriched_df, ENRICHED_DIR / "networking_papers_enriched.parquet")

    print_coverage_report(enriched_df)

    print("\n--------------------")
    print("STEP 3: Embeddings")
    print("--------------------\n")

    enriched_df = add_embeddings(enriched_df)
    save_dataframe(enriched_df, ENRICHED_DIR / "networking_papers_enriched.parquet")


    print("\n--------------------")
    print("STEP 4: Chunking")
    print("--------------------\n")

    # only chunk papers that have abstracts
    chunkable = enriched_df[enriched_df["abstract"].notna()].copy()
    print(f"Chunking {len(chunkable)}/{len(enriched_df)} papers with abstracts")

    chunks_df = create_chunks(chunkable)

    save_dataframe(chunks_df, CHUNKS_DIR / "networking_chunks.parquet")
    print(f"Total chunks: {len(chunks_df)}")

    print("\nDONE ✓")


if __name__ == "__main__":
    main()
