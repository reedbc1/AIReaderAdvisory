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
    """Embed the enhanced WR dataset and write FAISS index + NumPy matrix."""
    client = client or get_client()
    batch_size = 100

    print("\n📁 Choose dataset folder:")
    directory = prompt_for_subdirectory()

    # Always use this JSON filename
    json_path = os.path.join(directory, "wr_enhanced.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Could not find wr_enhanced.json in: {directory}")

    index_path = os.path.join(directory, "library.index")
    embeddings_path = os.path.join(directory, "library_embeddings.npy")

    print(f"\n📄 Loading records from {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    texts = [record_to_text(record) for record in records]
    all_embeddings: list[list[float] | None] = []

    for start in tqdm(range(0, len(texts), batch_size)):
        batch = texts[start:start + batch_size]
        embeddings = await embed_batch(batch, client=client)
        all_embeddings.extend(embeddings)
        await asyncio.sleep(0.1)

    valid_embeddings = [e for e in all_embeddings if e is not None]
    embedding_matrix = np.array(valid_embeddings).astype("float32")

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    faiss.write_index(index, index_path)
    np.save(embeddings_path, embedding_matrix)

    print("\n✅ Done. Saved FAISS index and embeddings.")
    print(f"Index → {index_path}")
    print(f"Embeddings → {embeddings_path}")


async def main():
    await embed_library()


if __name__ == "__main__":
    asyncio.run(main())
