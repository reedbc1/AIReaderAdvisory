"""
Vega catalog search helpers (format-group level only).

Responsibilities:
- Query Vega format-group search endpoint
- Partition-safe paging (avoid ES result window limits)
- Parse raw catalog records
- Write wr.json snapshots

This module intentionally does NOT:
- Fetch edition metadata
- Enrich records
- Manage state
- Generate embeddings
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm_asyncio

from choose_dir import replace_with_utf8_hex

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
    out_dir: Path,
    location_ids: Optional[int] = DEFAULT_LOCATION_IDS,
    material_type_ids: Optional[int] = DEFAULT_MATERIAL_TYPE_IDS,
) -> tuple[Path, int]:
    """
    Fetch one partition (e.g. 'A*') and write wr.json.

    Returns:
      (path_to_wr.json, total_results)
    """
    ensure_dir(out_dir)

    results_file = out_dir / "wr.json"
    info_file = out_dir / "info.json"

    # First page for metadata
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

    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "partition": partition_key,
                "totalPages": total_pages,
                "totalResults": total_results,
                "maxPagesFetched": max_pages,
            },
            f,
            indent=2,
        )

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("[\n")

    first = True

    def write_records(data: dict):
        nonlocal first
        for rec in parse_format_group_records(data.get("data", [])):
            if not first:
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(",\n")
            with open(results_file, "a", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            first = False

    # page 0
    write_records(first_data)

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
        if not data:
            continue
        write_records(data)

    with open(results_file, "a", encoding="utf-8") as f:
        f.write("\n]\n")

    return results_file, total_results

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
    "fetch_partition",
    "build_search_payload",
    "parse_format_group_records",
]
