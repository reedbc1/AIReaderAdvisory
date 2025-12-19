"""
Stateful catalog pipeline for the WR dataset.

Responsibilities:
- Diff canonical catalog snapshot vs stored state
- Enrich new/changed records (editions, metadata)
- Maintain wr_state.json
- Produce wr_enhanced.json (no embeddings)

Assumptions:
- Canonical snapshot already exists at:
    data/<RUN_ID>/wr.json
- Partitioning + merging handled upstream
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import aiohttp

# These now come from your ingestion layer
from catalog import HTTP_CONCURRENCY
from enrichment import process_record

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class DiffResult:
    new_records: List[dict]
    changed_records: List[dict]
    unchanged_records: List[dict]
    removed_ids: List[str]


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"records": [], "needs_index_rebuild": False}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def source_hash(record: dict) -> str:
    """
    Stable hash of catalog fields *prior* to embedding.
    This intentionally ignores runtime-only fields.
    """
    clean = {
        k: v
        for k, v in record.items()
        if k not in {"embedded", "embedding"}
    }
    return hashlib.md5(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def build_lookup(records: Iterable[dict]) -> Dict[str, dict]:
    return {r["id"]: r for r in records if r.get("id")}


def diff_catalog_records(
    catalog_records: List[dict],
    stored_records: List[dict],
) -> DiffResult:
    stored_lookup = build_lookup(stored_records)
    catalog_lookup = build_lookup(catalog_records)

    new_records: list[dict] = []
    changed_records: list[dict] = []
    unchanged_records: list[dict] = []

    for record in catalog_records:
        rid = record.get("id")
        if not rid:
            continue

        record_hash = source_hash(record)
        existing = stored_lookup.get(rid)

        if existing is None:
            new_records.append({**record, "source_hash": record_hash})
        elif existing.get("source_hash") != record_hash:
            changed_records.append({**record, "source_hash": record_hash})
        else:
            unchanged_records.append(existing)

    removed_ids = [
        rid for rid in stored_lookup.keys()
        if rid not in catalog_lookup
    ]

    return DiffResult(
        new_records=new_records,
        changed_records=changed_records,
        unchanged_records=unchanged_records,
        removed_ids=removed_ids,
    )


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

async def enrich_records(records: List[dict]) -> List[dict]:
    """
    Enrich records with edition metadata (subjects, summary, contributors).
    """
    if not records:
        return []

    semaphore = asyncio.Semaphore(HTTP_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_record(record, session, semaphore)
            for record in records
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def load_catalog_snapshot(path: str) -> List[dict]:
    """
    Load canonical catalog snapshot (already merged from partitions).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog snapshot not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def drop_runtime_fields(record: dict) -> dict:
    """
    Remove runtime-only fields so wr_enhanced.json stays clean.
    """
    return {
        k: v
        for k, v in record.items()
        if k not in {"embedded", "embedding", "source_hash"}
    }


def write_enhanced_snapshot(records: List[dict], path: str) -> None:
    clean = [drop_runtime_fields(r) for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def sync_catalog_state(
    *,
    run_dir: str,
) -> dict:
    """
    Incrementally sync canonical catalog snapshot into local state.

    Expects:
      run_dir/
        wr.json
        wr_state.json
        wr_enhanced.json
    """
    snapshot_path = os.path.join(run_dir, "wr.json")
    state_path = os.path.join(run_dir, "wr_state.json")
    enhanced_path = os.path.join(run_dir, "wr_enhanced.json")

    catalog_records = load_catalog_snapshot(snapshot_path)
    state = load_state(state_path)
    stored_records: list[dict] = state.get("records", [])

    diff = diff_catalog_records(catalog_records, stored_records)

    # Enrich only new + changed
    to_enrich = diff.new_records + diff.changed_records
    enriched = await enrich_records(to_enrich)

    updated_records: list[dict] = []
    updated_records.extend(diff.unchanged_records)

    incoming_lookup = build_lookup(diff.new_records + diff.changed_records)

    for record in enriched:
        rid = record.get("id")
        base = incoming_lookup.get(rid, {})
        merged = {
            **record,
            "embedded": False,
            "embedding": None,
            **base,
        }
        updated_records.append(merged)

    # Remove deleted items
    updated_records = [
        r for r in updated_records
        if r.get("id") not in diff.removed_ids
    ]

    needs_index_rebuild = bool(
        diff.new_records or diff.changed_records or diff.removed_ids
    )

    state_out = {
        "records": sorted(updated_records, key=lambda r: r.get("id", "")),
        "needs_index_rebuild": needs_index_rebuild,
    }

    save_state(state_out, state_path)
    write_enhanced_snapshot(state_out["records"], enhanced_path)

    return {
        "new": len(diff.new_records),
        "changed": len(diff.changed_records),
        "unchanged": len(diff.unchanged_records),
        "removed": len(diff.removed_ids),
        "needs_index_rebuild": needs_index_rebuild,
    }


__all__ = [
    "DiffResult",
    "diff_catalog_records",
    "enrich_records",
    "sync_catalog_state",
]
