"""Interactive conversation loop for the WR recommendation agent."""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from choose_dir import prompt_for_subdirectory


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Lazily load environment variables and return the OpenAI client."""

    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_tools() -> list[dict[str, Any]]:
    """Tools available to the conversation model."""

    return [{
        "type": "web_search",
    }, {
        "type": "function",
        "name": "search_library",
        "description":
        "Find movies for the user based on what they say they are looking for.",
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
                    "description":
                    "The number of top similar results to return.",
                    "default": 5
                }
            },
            "required": ["query", "k"],
            "additionalProperties": False
        }
    }]


def load_library() -> Tuple[faiss.Index, list[dict]]:
    """Load FAISS index and JSON records for interactive search."""

    directory = prompt_for_subdirectory()

    index_path: str = f"{directory}/library.index"
    json_path: str = f"{directory}/wr_enhanced.json"
    embeddings_path: str = f"{directory}/library_embeddings.npy"

    index = faiss.read_index(index_path)
    with open(json_path, encoding="utf-8") as file:
        records = json.load(file)

    embeddings = np.load(embeddings_path)
    assert len(records) == embeddings.shape[
        0], "❌ Mismatch between JSON records and embeddings!"
    return index, records


def search_library(query: str,
                   k: int,
                   index: faiss.Index,
                   records: list[dict],
                   *,
                   client: OpenAI | None = None) -> List[Dict[str, Any]]:
    """Return top-k most similar library items to a text query."""

    client = client or get_client()
    response = client.embeddings.create(model="text-embedding-3-small",
                                        input=query)
    query_vec = np.array(response.data[0].embedding,
                         dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        record = records[idx]
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


def call_function(name: str, args: dict[str, Any], *, index: faiss.Index,
                  records: list[dict], client: OpenAI) -> list[dict[str, Any]]:
    """Dispatch available function tools."""

    if name == "search_library":
        return search_library(**args,
                              index=index,
                              records=records,
                              client=client)
    raise ValueError(f"Unknown tool: {name}")


def create_conversation(*, client: OpenAI | None = None) -> str:
    """Create a new conversation and return its id."""

    client = client or get_client()
    conversation = client.conversations.create()
    return conversation.id


def run_conversation_loop():
    client = get_client()
    tools = get_tools()
    index, records = load_library()
    conv_id = create_conversation(client=client)

    while True:
        query = str(input("Enter query (or 'exit' to quit): "))
        if query.lower() == "exit":
            break

        input_messages = [{"role": "user", "content": f"{query}"}]
        response = client.responses.create(model="gpt-5",
                                           tools=tools,
                                           input=input_messages,
                                           conversation=conv_id)

        for tool_call in response.output or []:
            if tool_call.type != "function_call":
                continue

            name = tool_call.name
            args = json.loads(tool_call.arguments)
            result = call_function(name,
                                   args,
                                   index=index,
                                   records=records,
                                   client=client)

            client.conversations.items.create(conv_id,
                                              items=[{
                                                  "type":
                                                  "function_call_output",
                                                  "call_id":
                                                  tool_call.call_id,
                                                  "output":
                                                  json.dumps(result)
                                              }])

            response = client.responses.create(
                model="gpt-5",
                input=
                "Pick three of the movies generated by a tool and explain how they match the query.",
                conversation=conv_id)

        print(response.output_text)


if __name__ == "__main__":
    run_conversation_loop()
