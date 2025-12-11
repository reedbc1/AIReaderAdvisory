import asyncio
import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def record_to_text(record):
    """Convert a record from the enhanced WR dataset into a promptable string."""

    return (
        f"Title: {record['title']}\n"
        f"Author: {record.get('author', '')}\n"
        f"Material: {record.get('materials', '')[0].get('name')}\n"
        f"Publication Date: {record.get('publicationDate', '')}\n"
        f"Contributors: {record.get('contributors', '')}\n"
        f"Subjects: {record.get('subjects', '')}\n"
        f"Description: {record.get('summary', '')}"
    )


async def embed_batch(batch, *, retries=5, pause_seconds=0.1):
    """Embed a batch of texts with basic exponential backoff."""

    for attempt in range(retries):
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            return [item.embedding for item in response.data]
        except Exception as exc:  # pragma: no cover - network error handling
            wait_time = 2 ** attempt
            print(f"Error: {exc} — retrying in {wait_time} sec")
            await asyncio.sleep(wait_time)

    # fallback if all retries fail
    await asyncio.sleep(pause_seconds)
    return [None] * len(batch)


async def embed_library(
    json_path="json_files/wr_enhanced.json",
    *,
    batch_size=100,
    index_path="library.index",
    embeddings_path="library_embeddings.npy",
):
    """Embed the enhanced WR dataset and write FAISS index + numpy matrix."""

    with open(json_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    texts = [record_to_text(record) for record in records]
    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size)):
        batch = texts[start:start + batch_size]
        embeddings = await embed_batch(batch)
        all_embeddings.extend(embeddings)
        # small pause to respect rate limits
        await asyncio.sleep(0.1)

    valid_embeddings = [embedding for embedding in all_embeddings if embedding is not None]
    embedding_matrix = np.array(valid_embeddings).astype("float32")

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    faiss.write_index(index, index_path)
    np.save(embeddings_path, embedding_matrix)
    print("✅ Done. Saved FAISS index and embeddings.")


async def main():
    await embed_library()


if __name__ == "__main__":
    asyncio.run(main())
