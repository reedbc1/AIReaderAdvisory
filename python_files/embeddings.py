"""Embedding helpers for the WR dataset."""

import asyncio
import json
import os
from functools import lru_cache
from typing import Iterable, List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from choose_dir import prompt_for_subdirectory
from stateful_pipeline import STATE_FILE, load_state, save_state


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    """Lazily load environment variables and return the OpenAI client."""
    load_dotenv()
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def record_to_text(record: dict) -> str:
    """Convert an enriched WR record into a prompt-ready string."""
    materials = record.get("materials") or [{}]
    material_name = materials[0].get("name", "")

    return (f"Title: {record.get('title', '')}\n"
            f"Author: {record.get('author', '')}\n"
            f"Material: {material_name}\n"
            f"Publication Date: {record.get('publicationDate', '')}\n"
            f"Contributors: {record.get('contributors', '')}\n"
            f"Subjects: {record.get('subjects', '')}\n"
            f"Description: {record.get('summary', '')}")


async def embed_batch(
        batch: Iterable[str],
        *,
        retries: int = 5,
        pause_seconds: float = 0.1,
        client: AsyncOpenAI | None = None) -> List[List[float] | None]:
    """Embed a batch of texts with exponential backoff."""
    client = client or get_client()
    batch_list = list(batch)

    for attempt in range(retries):
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small", input=batch_list)
            return [item.embedding for item in response.data]
        except Exception as exc:
            wait_time = 2**attempt
            print(f"Error: {exc} — retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    print("❌ All retries failed — returning None embeddings.")
    return [None] * len(batch_list)


async def embed_library(client: AsyncOpenAI | None = None):
    """Embed new/changed items and rebuild the FAISS index when required."""
    client = client or get_client()
    batch_size = 100

    print("\n📁 Choose dataset folder:")
    directory = prompt_for_subdirectory()

    json_path = os.path.join(directory, "wr_enhanced.json")
    index_path = os.path.join(directory, "library.index")
    embeddings_path = os.path.join(directory, "library_embeddings.npy")

    state_path = os.path.join(directory, os.path.basename(STATE_FILE))
    state = load_state(state_path)
    records = state.get("records", [])

    if not records and os.path.exists(json_path):
        # Backwards compatibility: load legacy enhanced file when no state is present
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        state = {"records": records, "needs_index_rebuild": True}

    # Ensure in-memory list is attached to state for persistence
    state["records"] = records

    pending = [r for r in records if not r.get("embedded")]
    if not pending:
        print("\nℹ️ No records require embedding.")
    else:
        print(f"\n📄 Embedding {len(pending)} new/updated records ...")
        texts = [record_to_text(record) for record in pending]
        all_embeddings: list[list[float] | None] = []

        for start in tqdm(range(0, len(texts), batch_size)):
            batch = texts[start:start + batch_size]
            embeddings = await embed_batch(batch, client=client)
            all_embeddings.extend(embeddings)
            await asyncio.sleep(0.1)

        for record, emb in zip(pending, all_embeddings):
            record["embedded"] = emb is not None
            record["embedding"] = emb

        state["needs_index_rebuild"] = True

    if state.get("needs_index_rebuild"):
        embedded_records = [r for r in records if r.get("embedded")]
        if not embedded_records:
            print("\n⚠️ No embedded records available; skipping index rebuild.")
        else:
            embedding_matrix = np.array([r.get("embedding") for r in embedded_records],
                                        dtype="float32")
            index = faiss.IndexFlatL2(embedding_matrix.shape[1])
            index.add(embedding_matrix)

            # Persist metadata for downstream consumers
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([{
                    k: v
                    for k, v in record.items()
                    if k not in {"embedded", "embedding", "source_hash"}
                } for record in embedded_records], f, ensure_ascii=False, indent=2)

            faiss.write_index(index, index_path)
            np.save(embeddings_path, embedding_matrix)

            state["needs_index_rebuild"] = False
            print("\n✅ Rebuilt FAISS index and embeddings.")
    else:
        print("\nℹ️ Skipping index rebuild (no changes detected).")

    save_state(state, state_path)
    print(f"State saved → {state_path}")


async def main():
    await embed_library()


if __name__ == "__main__":
    asyncio.run(main())
