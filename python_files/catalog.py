"""
Lightweight helpers for fetching and enriching Vega catalog data.

Responsibilities:
- Query Vega format-group search endpoint across partitions
- Parse records into a minimal shape
- Enrich records with edition metadata (subjects, summary, contributors)
- Persist wr.json and wr_enhanced.json inside a run directory
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import aiohttp
from tqdm.asyncio import tqdm_asyncio

# ---------------------------------------------------------------------
# Vega endpoints
# ---------------------------------------------------------------------

BASE_SEARCH_URL = "https://na2.iiivega.com/api/search-result/search/format-groups"
BASE_EDITION_URL = "https://na2.iiivega.com/api/search-result/editions"  # used by enrichment only

# ---------------------------------------------------------------------
# HTTP / paging config
# ---------------------------------------------------------------------

HTTP_CONCURRENCY = 6
PAGE_SIZE = 1000
MAX_ES_RESULTS = 10_000
MAX_ES_PAGES = MAX_ES_RESULTS // PAGE_SIZE

REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 5

# ---------------------------------------------------------------------
# Query parameters (override from caller if needed)
# ---------------------------------------------------------------------

DEFAULT_LOCATION_IDS = 59          # Weber Road
DEFAULT_MATERIAL_TYPE_IDS = None   # e.g. 33 for DVDs

# ---------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "api-version": "2",
    "iii-customer-domain": "slouc.na2.iiivega.com",
    "iii-host-domain": "slouc.na2.iiivega.com",
    "origin": "https://slouc.na2.iiivega.com",
    "referer": "https://slouc.na2.iiivega.com/",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------

def _is_retryable(status: int) -> bool:
    return status in {408, 429, 500, 502, 503, 504}

async def post_json_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    *,
    timeout_s: int = REQUEST_TIMEOUT_S,
    retries: int = MAX_RETRIES,
) -> Optional[dict]:
    attempt = 0

    while True:
        attempt += 1
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.json()

                text = await resp.text()
                if not _is_retryable(resp.status):
                    print(f"❌ Non-retryable HTTP {resp.status}: {text[:200]}")
                    return None

                if attempt > retries:
                    print(f"❌ Exhausted retries ({resp.status}): {text[:200]}")
                    return None

                delay = min(30, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                print(f"⚠️ HTTP {resp.status} — retrying in {delay:.1f}s")
                await asyncio.sleep(delay)

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt > retries:
                print(f"❌ Network failure after retries: {exc}")
                return None

            delay = min(30, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            print(f"⚠️ Network error ({exc}) — retrying in {delay:.1f}s")
            await asyncio.sleep(delay)

# ---------------------------------------------------------------------
# Vega payload & parsing
# ---------------------------------------------------------------------

def build_search_payload(
    *,
    partition_key: str,
    page_num: int,
    location_ids: Optional[int] = DEFAULT_LOCATION_IDS,
    material_type_ids: Optional[int] = DEFAULT_MATERIAL_TYPE_IDS,
) -> dict:
    payload = {
        "searchText": partition_key,
        "sorting": "title",
        "sortOrder": "asc",
        "searchType": "everything",
        "universalLimiterIds": ["at_library"],
        "pageNum": page_num,
        "pageSize": PAGE_SIZE,
        "resourceType": "FormatGroup",
    }

    if location_ids:
        payload["locationIds"] = location_ids
    if material_type_ids:
        payload["materialTypeIds"] = material_type_ids

    return payload


def parse_format_group_records(records: list[dict]) -> list[dict]:
    """
    Parse raw Vega format-group records into a stable, minimal structure.
    """
    parsed: list[dict] = []

    for r in records:
        parsed.append({
            "id": r.get("id"),
            "title": r.get("title"),
            "publicationDate": r.get("publicationDate"),
            "author": r.get("primaryAgent", {}).get("label"),
            "materials": [
                {
                    "name": m.get("name"),
                    "type": m.get("type"),
                    "callNumber": m.get("callNumber"),
                    "editions": [
                        {
                            "id": e.get("id"),
                            "publicationDate": e.get("publicationDate"),
                        }
                        for e in m.get("editions", [])
                    ],
                }
                for m in r.get("materialTabs", [])
            ],
        })

    return parsed

# ---------------------------------------------------------------------
# Partition fetch
# ---------------------------------------------------------------------

async def fetch_partition(
    *,
    session: aiohttp.ClientSession,
    partition_key: str,
    location_ids: Optional[int] = DEFAULT_LOCATION_IDS,
    material_type_ids: Optional[int] = DEFAULT_MATERIAL_TYPE_IDS,
) -> tuple[list[dict], int]:
    """
    Fetch one partition (e.g. 'A*') and return parsed records.

    Returns:
      (records, total_results_reported_by_api)
    """

    payload = build_search_payload(
        partition_key=partition_key,
        page_num=0,
        location_ids=location_ids,
        material_type_ids=material_type_ids,
    )

    first_data = await post_json_with_retries(session, BASE_SEARCH_URL, payload)
    if not first_data:
        raise RuntimeError(f"Failed to fetch first page for {partition_key}")

    total_pages = int(first_data.get("totalPages", 1))
    total_results = int(first_data.get("totalResults", 0))
    max_pages = min(total_pages, MAX_ES_PAGES)

    records: list[dict] = []

    def collect(data: dict) -> None:
        records.extend(parse_format_group_records(data.get("data", [])))

    collect(first_data)

    for page_num in tqdm_asyncio(
        range(1, max_pages),
        desc=f"📜 {partition_key}",
    ):
        payload = build_search_payload(
            partition_key=partition_key,
            page_num=page_num,
            location_ids=location_ids,
            material_type_ids=material_type_ids,
        )
        data = await post_json_with_retries(session, BASE_SEARCH_URL, payload)
        if data:
            collect(data)

    return records, total_results


# ---------------------------------------------------------------------
# Edition enrichment
# ---------------------------------------------------------------------

async def fetch_edition(session: aiohttp.ClientSession, edition_id: str) -> dict:
    url = f"{BASE_EDITION_URL}/{edition_id}"
    async with session.get(url) as response:
        if response.status != 200:
            return {}
        return await response.json()


def process_edition(edition: dict) -> dict:
    data = edition.get("edition", {}) if isinstance(edition, dict) else {}

    subjects = "; ".join(
        v for k, v in data.items()
        if isinstance(k, str) and k.startswith("subj") and isinstance(v, str)
    )

    notes = " ".join(
        v for k, v in data.items()
        if isinstance(k, str) and k.startswith("note") and isinstance(v, str)
    )

    return {
        "subjects": subjects,
        "summary": notes,
        "contributors": data.get("contributors", []),
    }


async def enrich_record(
    record: dict,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> dict:
    materials = record.get("materials") or []
    edition_id = None

    if materials:
        editions = materials[0].get("editions") or []
        if editions:
            edition_id = editions[0].get("id")

    if not edition_id:
        return record

    async with semaphore:
        edition_info = await fetch_edition(session, edition_id)

    record.update(process_edition(edition_info))
    return record


async def enrich_records(records: list[dict]) -> list[dict]:
    if not records:
        return []

    semaphore = asyncio.Semaphore(HTTP_CONCURRENCY)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = [
            enrich_record(record, session, semaphore)
            for record in records
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------
# Catalog orchestration
# ---------------------------------------------------------------------

DEFAULT_PARTITIONS = [f"{chr(c)}*" for c in range(ord("A"), ord("Z") + 1)] + ["0*"]


async def fetch_catalog_snapshot(
    *,
    run_dir: Path,
    partitions: Iterable[str] = DEFAULT_PARTITIONS,
    location_ids: Optional[int] = DEFAULT_LOCATION_IDS,
    material_type_ids: Optional[int] = DEFAULT_MATERIAL_TYPE_IDS,
) -> dict:
    """
    Fetch partitions, merge into wr.json, and return a summary.
    """
    ensure_dir(run_dir)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = [
            fetch_partition(
                session=session,
                partition_key=partition_key,
                location_ids=location_ids,
                material_type_ids=material_type_ids,
            )
            for partition_key in partitions
        ]
        results = await asyncio.gather(*tasks)

    merged_records: list[dict] = []
    reported_total = 0

    for records, total in results:
        merged_records.extend(records)
        reported_total += total

    snapshot_path = run_dir / "wr.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, ensure_ascii=False, indent=2)

    return {
        "snapshot": str(snapshot_path),
        "records_written": len(merged_records),
        "reported_total": reported_total,
    }


async def enrich_snapshot(run_dir: Path) -> Path:
    """
    Load wr.json, enrich edition metadata, and write wr_enhanced.json.
    """
    snapshot_path = run_dir / "wr.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Catalog snapshot not found: {snapshot_path}")

    with open(snapshot_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"📚 Enriching {len(records)} records…")
    enriched = await enrich_records(records)

    enhanced_path = run_dir / "wr_enhanced.json"
    with open(enhanced_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    return enhanced_path


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "BASE_SEARCH_URL",
    "BASE_EDITION_URL",
    "HTTP_CONCURRENCY",
    "PAGE_SIZE",
    "MAX_ES_RESULTS",
    "DEFAULT_LOCATION_IDS",
    "DEFAULT_MATERIAL_TYPE_IDS",
    "DEFAULT_PARTITIONS",
    "fetch_catalog_snapshot",
    "enrich_snapshot",
    "fetch_partition",
    "build_search_payload",
    "parse_format_group_records",
]
