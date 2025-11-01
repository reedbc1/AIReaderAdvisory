import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conversation = client.conversations.create(metadata={"topic": "demo"},
                                           items=[{
                                               "type": "message",
                                               "role": "user",
                                               "content": "Hello!"
                                           }])

conv_id = conversation.id

# Load index and data
index = faiss.read_index("library.index")

with open("json_files/wr_enhanced.json", encoding="utf-8") as f:
    data = json.load(f)

embeddings = np.load("library_embeddings.npy")
assert len(data) == embeddings.shape[
    0], "❌ Mismatch between JSON records and embeddings!"

# 1. Define a list of callable tools for the model
tools = [{
    "type": "function",
    "name": "search_library",
    "description": "Find similar movies to user request.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type":
                "string",
                "description":
                ("A description of what kinds of movies the customer is looking for,"
                 "including the names of movies and actors.")
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}]


def search_library(input):
    """Return top-k most similar library items to a text query."""
    query = input.get("query")
    k = 5

    print("search_library ran!")

    # Get embedding for the query
    response = client.embeddings.create(model="text-embedding-3-small",
                                        input=query)
    query_vec = np.array(response.data[0].embedding,
                         dtype="float32").reshape(1, -1)

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


items = client.conversations.items.list(conv_id, limit=10)
print("printing items.data...")
print(items.data)

# Create a running input list we will add to over time
query = "I'm looking for movies like pans labrynth"

# 2. Prompt the model with tools defined
response = client.responses.create(model="gpt-5",
                                   tools=tools,
                                   tool_choice="required",
                                   input=query,
                                   conversation=conv_id)

print(response.output_text)

response = client.responses.create(
    input=query,
    model="gpt-5",
    instructions="Pick 3 movies relevant to the query from the tool output.",
    tools=tools,
    conversation=conv_id,
)

print(response.output_text)
