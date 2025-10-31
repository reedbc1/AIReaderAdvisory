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

with open("json_files/wr_enhanced.json", encoding="utf-8") as f:
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
            "material": record.get("materials", [])[0].get("name"),
            "year": record.get("publicationDate"),
            "summary": record.get("summary"),
            "subjects": record.get("subjects"),
            "contributors": record.get("contributors"),
            "distance": float(dist)
        })
    return results


def recommend_library_items(query, k=10):
    candidates = search_library(query, k)
    prompt = (
        f"The user is looking for: '{query}'.\n\n"
        "Here are some possible matches:\n\n" +
        "\n".join([f"{i+1}. {c['title']} — {c['summary']}" for i, c in enumerate(candidates)]) +
        "\n\nRank the best 3 recommendations and explain why."
    )

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    print(response.output_text)


def top_results(query):
    results = search_library(query)

    for r in results:
        print(f"📖 {r['title']} ({r['year']}) by {r['author']}")
        print(f"   Material: {r['material']}")
        print(f"   Subjects: {r['subjects']}")
        print(f"   Summary: {r['summary'][:200]}...")
        print(f"   Contributors: {r['contributors']}")
        print(f"   Similarity score: {r['distance']:.4f}")
        print()


def lil_guy(query):
    
    response = client.responses.create(
        model="gpt-5-mini",
        input=query,
        conversation="conv_69040240320081948d0927724758d2dc0e28cd9880c6665f",
        instructions="you're just a lil guy"
    )

    print(response.output_text)


if __name__ == "__main__":
    # recommend_library_items("sexy sci fi movies")
    
    # lil_guy("aaachooo choooo!!!")
    conversation = client.conversations.retrieve("conv_69040240320081948d0927724758d2dc0e28cd9880c6665f")

    items = client.conversations.items.list(conversation.id, limit=10)
    print(items.data)

    