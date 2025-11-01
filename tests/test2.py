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
            },
            "k": {
                "type": "integer",
                "description": "The number of top similar results to return.",
                "default": 5
            }
        },
        "required": ["query", "k"],
        "additionalProperties": False
    }
}]


def search_library(query, k):
    """Return top-k most similar library items to a text query."""
    k = 5

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


def call_function(name, args):
    if name == "search_library":
        return search_library(**args)


# Create a running input list we will add to over time
query = "I'm looking for movies like pans labrynth"

input_messages = [{"role": "user", "content": f"{query}"}]

# 2. Prompt the model with tools defined
response = client.responses.create(
    model="gpt-5",
    tools=tools,
    tool_choice="required",
    input=input_messages,
)

for tool_call in response.output:
    if tool_call.type != "function_call":
        continue

    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)
    input_messages.append({"role": "user", "content": str(result)})

print(input_messages)
print("\n")

response = client.responses.create(model="gpt-5",
                                   input=input_messages,
                                   tools=tools)

print(response.output_text)
