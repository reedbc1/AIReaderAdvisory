"""
Embedding helpers for the WR dataset (canonical, stateful version).

Assumes directory layout created by the partitioned Vega ingestion pipeline:

data/<RUN_ID>/
├─ wr_enhanced.json        # canonical enriched snapshot
├─ wr_state.json           # state file (records + needs_index_rebuild)
├─ library.index           # FAISS index
└─ library_embeddings.npy  # embedding matrix
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from typing import Iterable, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from stateful_pipeline import STATE_FILE, load_state, save_state


# ---------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    load_dotenv()
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------
# Record → embedding text
# ---------------------------------------------------------------------

def record_to_text(record: dict) -> str:
    materials = record.get("materials") or [{}]
    material_name = materials[0].get("name", "")

    return (
        f"Title: {record.get('title', '')}\n"
        f"Author: {record.get('author', '')}\n"
        f"Material: {material_name}\n"
        f"Publication Date: {record.get('publicationDate', '')}\n"
        f"Contributors: {record.get('contributors', '')}\n"
        f"Subjects: {record.get('subjects', '')}\n"
        f"Description: {record.get('summary', '')}"
    )


# ---------------------------------------------------------------------
# Per-record embedding with retries (Gemini-safe pattern)
# ---------------------------------------------------------------------

async def embed_one(
    text: str,
    *,
    client: AsyncOpenAI,
    retries: int = 5,
) -> Optional[list[float]]:
    for attempt in range(retries):
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            wait = min(30, 2 ** attempt)
            print(f"⚠️ Embedding error ({exc}) — retrying in {wait}s")
            await asyncio.sleep(wait)

    print("❌ Embedding failed after retries.")
    return None


# ---------------------------------------------------------------------
# Main embedding orchestration
# ---------------------------------------------------------------------

async def embed_library(run_dir: str) -> None:
    """
    Embed new/changed records and rebuild FAISS index if required.

    run_dir: e.g. data/*_None_59
    """
    client = get_client()

    json_path = os.path.join(run_dir, "wr_enhanced.json")
    index_path = os.path.join(run_dir, "library.index")
    embeddings_path = os.path.join(run_dir, "library_embeddings.npy")

    state_path = os.path.join(run_dir, os.path.basename(STATE_FILE))
    state = load_state(state_path)

    records: list[dict] = state.get("records", [])

    if not records:
        raise RuntimeError("State file contains no records — cannot embed.")

    pending = [r for r in records if not r.get("embedded")]

    if not pending:
        print("ℹ️ No records require embedding.")
    else:
        print(f"📄 Embedding {len(pending)} records…")

        for record in tqdm(pending, desc="🧠 Embedding"):
            text = record_to_text(record)
            embedding = await embed_one(text, client=client)

            record["embedded"] = embedding is not None
            record["embedding"] = embedding

        state["needs_index_rebuild"] = True

    # -----------------------------------------------------------------
    # FAISS rebuild (only when needed)
    # -----------------------------------------------------------------

    if state.get("needs_index_rebuild"):
        embedded = [r for r in records if r.get("embedded")]

        if not embedded:
            print("⚠️ No embedded records — skipping index rebuild.")
        else:
            print("🔧 Rebuilding FAISS index…")

            matrix = np.array(
                [r["embedding"] for r in embedded],
                dtype="float32",
            )

            index = faiss.IndexFlatL2(matrix.shape[1])
            index.add(matrix)

            faiss.write_index(index, index_path)
            np.save(embeddings_path, matrix)

            # Persist clean enhanced snapshot (no embeddings)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            k: v
                            for k, v in r.items()
                            if k not in {"embedded", "embedding", "source_hash"}
                        }
                        for r in embedded
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            state["needs_index_rebuild"] = False
            print("✅ FAISS index rebuilt.")

    else:
        print("ℹ️ Index rebuild not required.")

    save_state(state, state_path)
    print(f"💾 State saved → {state_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

async def main():
    """
    Example:
      python embeddings_pipeline.py data/*_None_59
    """
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python embeddings_pipeline.py <run_dir>")

    await embed_library(sys.argv[1])


if __name__ == "__main__":
    asyncio.run(main())
