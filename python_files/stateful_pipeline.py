"""Incremental, state-aware catalog ingestion helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import aiohttp

from catalog import (
    CONCURRENCY,
    ENHANCED_FILE,
    RESULTS_FILE,
    create_dir,
    directory_name,
    process_record,
    vega_search,
)

STATE_FILE = f"{directory_name}/wr_state.json"


@dataclass
class DiffResult:
    new_records: List[dict]
    changed_records: List[dict]
    unchanged_records: List[dict]
    removed_ids: List[str]


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state(path: str = STATE_FILE) -> dict:
    """Load the persisted state file if present."""
    if not os.path.exists(path):
        return {"records": [], "needs_index_rebuild": False}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict, path: str = STATE_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def source_hash(record: dict) -> str:
    """Stable hash of the catalog-provided fields (pre-enrichment)."""
    return hashlib.md5(json.dumps(record, sort_keys=True).encode()).hexdigest()


def build_lookup(records: Iterable[dict]) -> Dict[str, dict]:
    return {r["id"]: r for r in records if "id" in r}


def diff_catalog_records(
    catalog_records: List[dict], stored_records: List[dict]
) -> DiffResult:
    """Identify new/changed/unchanged/removed items by stable ID."""
    stored_lookup = build_lookup(stored_records)
    catalog_lookup = build_lookup(catalog_records)

    new_records: list[dict] = []
    changed_records: list[dict] = []
    unchanged_records: list[dict] = []

    for record in catalog_records:
        record_id = record.get("id")
        if record_id is None:
            continue

        existing = stored_lookup.get(record_id)
        record_hash = source_hash(record)

        if existing is None:
            new_records.append({**record, "source_hash": record_hash})
        elif existing.get("source_hash") != record_hash:
            changed_records.append({**record, "source_hash": record_hash})
        else:
            unchanged_records.append(existing)

    removed_ids = [rid for rid in stored_lookup.keys() if rid not in catalog_lookup]
    return DiffResult(new_records, changed_records, unchanged_records, removed_ids)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

async def enrich_records(records: List[dict]) -> List[dict]:
    if not records:
        return []

    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [process_record(record, session, semaphore) for record in records]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Sync orchestration
# ---------------------------------------------------------------------------

async def load_catalog_snapshot() -> List[dict]:
    """Fetch the latest catalog snapshot and return parsed records."""
    await create_dir()
    await vega_search()
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _drop_runtime_fields(record: dict) -> dict:
    """Remove embedding metadata so wr_enhanced stays clean."""
    return {
        k: v
        for k, v in record.items()
        if k not in {"embedded", "embedding", "source_hash"}
    }


def write_enhanced_snapshot(records: List[dict]) -> None:
    """Persist the enriched records without embeddings for downstream use."""
    clean_records = [_drop_runtime_fields(r) for r in records]
    with open(ENHANCED_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)


async def sync_catalog_state() -> dict:
    """Incrementally sync catalog → local state."""
    catalog_records = await load_catalog_snapshot()
    state = load_state()
    stored_records: list[dict] = state.get("records", [])

    diff = diff_catalog_records(catalog_records, stored_records)

    # Enrich new + changed
    to_enrich = diff.new_records + diff.changed_records
    enriched = await enrich_records(to_enrich)

    updated_records: list[dict] = []
    # Keep unchanged as-is
    updated_records.extend(diff.unchanged_records)

    # Apply new/changed with metadata flags
    incoming_lookup = build_lookup(diff.new_records + diff.changed_records)
    for record in enriched:
        rid = record.get("id")
        base = incoming_lookup.get(rid, {})
        merged = {**record, "embedded": False, "embedding": None, **base}
        updated_records.append(merged)

    # Preserve ids only present in stored when unchanged/updated
    updated_records = [r for r in updated_records if r.get("id") not in diff.removed_ids]

    needs_index_rebuild = state.get("needs_index_rebuild", False)
    if diff.removed_ids or diff.new_records or diff.changed_records:
        needs_index_rebuild = True

    state = {
        "records": sorted(updated_records, key=lambda r: r.get("id", "")),
        "needs_index_rebuild": needs_index_rebuild,
    }

    save_state(state)
    write_enhanced_snapshot(state["records"])

    return {
        "new": len(diff.new_records),
        "changed": len(diff.changed_records),
        "unchanged": len(diff.unchanged_records),
        "removed": len(diff.removed_ids),
    }


__all__ = [
    "STATE_FILE",
    "diff_catalog_records",
    "enrich_records",
    "load_catalog_snapshot",
    "sync_catalog_state",
]
