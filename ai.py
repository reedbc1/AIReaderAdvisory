# purpose: create embeddings for vega results

import openai
import os
from dotenv import load_dotenv
import tiktoken
import faiss
import numpy as np
from tqdm import tqdm
import json
import asyncio
import json
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
import faiss
import time

load_dotenv()

# Load JSON
with open("json_files/wr_enhanced.json", "r", encoding="utf-8") as f:
    data = json.load(f)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def record_to_text(r):
    return (
        f"Title: {r['title']}\n"
        f"Author: {r.get('author', '')}\n"
        f"Material: {r.get('materials', '')[0].get('name')}\n"
        f"Publication Date: {r.get('publicationDate', '')}\n"
        f"Contributors: {r.get('contributors', '')}\n"
        f"Subjects: {r.get('subjects', '')}\n"
        f"Description: {r.get('summary', '')}"
    )

texts = [record_to_text(r) for r in data]
batch_size = 100


async def embed_batch(batch):
    # Retry with exponential backoff
    for attempt in range(5):
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            print(f"Error: {e} — retrying in {2 ** attempt} sec")
            await asyncio.sleep(2 ** attempt)
    return [None] * len(batch)  # fallback if all retries fail


async def main():
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embeddings = await embed_batch(batch)
        all_embeddings.extend(embeddings)
        # small pause to respect rate limits
        await asyncio.sleep(0.1)

    # Convert to numpy
    valid_embeddings = [e for e in all_embeddings if e is not None]
    embedding_matrix = np.array(valid_embeddings).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save
    faiss.write_index(index, "library.index")
    np.save("library_embeddings.npy", embedding_matrix)
    print("✅ Done. Saved FAISS index and embeddings.")

if __name__ == "__main__":
    asyncio.run(main())