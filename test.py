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

# 1. Define a list of callable tools for the model
tools = [
{
  "type": "function",
  "name": "search_library",
  "description": "Return top-k most similar library items to a text query.",
  "strict": True,
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": ("A description of what kinds of movies the customer is looking for,"
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
}
]


def search_library(arguments):
    """Return top-k most similar library items to a text query."""
    query = arguments.get("query")
    k = 5

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

# Create a running input list we will add to over time
query = "I'm looking for movies like pans labrynth"

input_list = [
    {"role": "user", "content": f"{query}"}
]

# 2. Prompt the model with tools defined
response = client.responses.create(
    model="gpt-5",
    tools=tools,
    tool_choice="required",
    input=input_list,
)

# Save function call outputs for subsequent requests
input_list += response.output

for item in response.output:
    if item.type == "function_call":
        if item.name == "search_library":
            # 3. Execute the function logic for search_library
            print("Arguments:\n")
            print(item.arguments)
            print("\n")
            print("Executing search_library function...")
            results = search_library(json.loads(item.arguments))
            
            # 4. Provide function call results to the model
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({
                  "results": results
                })
            })

print("Final input:")
print(input_list)
print("\n")

response = client.responses.create(
    model="gpt-5",
    instructions="Pick 3 relevant movies from the input list. Explain how they match the query.",
    input=input_list,
)

# 5. The model should be able to give a response!
print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)