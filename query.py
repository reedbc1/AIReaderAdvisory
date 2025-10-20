import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os


# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load index and data
index = faiss.read_index("library.index")

with open("json_files/wr_flat.json") as f:
    data = json.load(f)

embeddings = np.load("library_embeddings.npy")
assert len(data) == embeddings.shape[0], "❌ Mismatch between JSON records and embeddings!"

def search_library(query, k=5):
    """Return top-k most similar library items to a text query."""
    # Get embedding for the query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    
    # Search FAISS
    distances, indices = index.search(query_vec, k)
    
    # Return top results with metadata
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        record = data[idx]
        results.append({
            "title": record.get("title"),
            "author": record.get("author"),
            "material": record.get("material_name"),
            "type": record.get("material_type"),
            "year": record.get("publicationDate"),
            "notes": record.get("notes"),
            "subjects": record.get("subjects"),
            "distance": float(dist)
        })
    return results

query = "good action movie like the accountant"
results = search_library(query)
print(results)

for r in results:
    print(f"📖 {r['title']} ({r['year']}) by {r['author']}")
    print(f"   Material: {r['material']} [{r['type']}]")
    print(f"   Subjects: {r['subjects']}")
    print(f"   Notes: {r['notes'][:200]}...")
    print(f"   Similarity score: {r['distance']:.4f}")
    print()

def recommend_library_items(query, k=10):
    candidates = search_library(query, k)
    prompt = (
        f"The user is looking for: '{query}'.\n\n"
        "Here are some possible matches:\n\n" +
        "\n".join([f"{i+1}. {c['title']} — {c['notes']}" for i, c in enumerate(candidates)]) +
        "\n\nRank the best 3 recommendations and explain why."
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    print(response.output_text)
