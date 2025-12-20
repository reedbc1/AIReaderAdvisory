"""
Embedding helpers for the WR dataset.

Assumes a run directory that contains:
- wr_enhanced.json        # enriched snapshot (no embeddings)
- library.index           # FAISS index (written by this module)
- library_embeddings.npy  # embedding matrix (written by this module)
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm


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
    Embed all records from wr_enhanced.json and rebuild the FAISS index.
    """
    client = get_client()

    run_path = Path(run_dir)
    json_path = run_path / "wr_enhanced.json"
    index_path = run_path / "library.index"
    embeddings_path = run_path / "library_embeddings.npy"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing enriched snapshot: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        raise RuntimeError("No records found in wr_enhanced.json — cannot embed.")

    print(f"📄 Embedding {len(records)} records…")
    matrix: list[list[float]] = []

    for record in tqdm(records, desc="🧠 Embedding"):
        text = record_to_text(record)
        embedding = await embed_one(text, client=client)
        if embedding is None:
            raise RuntimeError("Embedding failed; aborting index build.")
        matrix.append(embedding)

    matrix_np = np.array(matrix, dtype="float32")
    index = faiss.IndexFlatL2(matrix_np.shape[1])
    index.add(matrix_np)

    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, matrix_np)

    print(f"✅ Saved {len(records)} embeddings to {embeddings_path}")
    print(f"✅ FAISS index written to {index_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

async def main():
    """
    Example:
      python embeddings.py data/current_run
    """
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python embeddings_pipeline.py <run_dir>")

    await embed_library(sys.argv[1])


if __name__ == "__main__":
    asyncio.run(main())
