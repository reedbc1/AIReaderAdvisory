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

def run_flatten():
    def flatten_library_data(data):
        flattened = []

        for record in data:
            id = record.get("id")
            title = record.get("title")
            author = record.get("author")
            publication_date = record.get("publicationDate")

            for material in record.get("materials", []):
                mat_name = material.get("name")
                mat_type = material.get("type")
                call_number = material.get("callNumber")

                for edition in material.get("editions", []):
                    flattened.append({
                        "record_id": id,
                        "edition_id": edition.get("id"),
                        "title": title,
                        "author": author,
                        "material_name": mat_name,
                        "material_type": mat_type,
                        "callNumber": call_number,
                        "publicationDate": edition.get("publicationDate", publication_date),
                        "contributors": edition.get("contributors"),
                        "notes": edition.get("notes"),
                        "subjects": edition.get("subjects"),
                    })

        return flattened
    
    with open("json_files/wr_enhanced.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    flattened_data = flatten_library_data(data)

    # Save as new file
    with open("json_files/wr_flat.json", "w") as f:
        json.dump(flattened_data, f, indent=2)

import json

# Load flattened JSON
with open("json_files/wr_enhanced.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Remove duplicates based on 'id'
seen_ids = set()
unique_data = []

for record in data:
    record_id = record.get("id")
    if record_id and record_id not in seen_ids:
        seen_ids.add(record_id)
        unique_data.append(record)

# Save back to JSON
with open("library_flat_unique.json", "w") as f:
    json.dump(unique_data, f, indent=2)

print(f"✅ Removed duplicates: {len(data) - len(unique_data)} duplicates dropped.")
print(f"Remaining records: {len(unique_data)}")

# Load flattened data
with open("library_flat_unique.json") as f:
    data = json.load(f)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def record_to_text(r):
    return (
        f"Title: {r['title']}\n"
        f"Author: {r.get('author', '')}\n"
        f"Material: {r.get('material_name', '')} ({r.get('material_type', '')})\n"
        f"Publication Date: {r.get('publicationDate', '')}\n"
        f"Contributors: {r.get('contributors', '')}\n"
        f"Subjects: {r.get('subjects', '')}\n"
        f"Description: {r.get('notes', '')}"
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
    # run_flatten()
    asyncio.run(main())
    







