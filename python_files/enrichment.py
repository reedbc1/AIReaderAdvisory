"""
Edition-level enrichment helpers for Vega catalog records.
"""

from __future__ import annotations

import asyncio
import aiohttp

from catalog import HTTP_CONCURRENCY, BASE_EDITION_URL


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


async def process_record(
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
