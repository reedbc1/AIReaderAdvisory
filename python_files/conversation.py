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


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------
# Optional web tool (placeholder)
# ---------------------------------------------------------------------

def get_web_tools() -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "name": "web_search",
        "description": "Look up recent or external context to refine a query.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }]


def run_web_search(query: str) -> dict:
    # Placeholder — plug in real web search here
    return {
        "summary": f"External context related to: {query}"
    }


# ---------------------------------------------------------------------
# Library loading + search
# ---------------------------------------------------------------------

def load_library() -> Tuple[faiss.Index, list[dict]]:
    directory = prompt_for_subdirectory()

    index_path = f"{directory}/library.index"
    json_path = f"{directory}/wr_enhanced.json"
    embeddings_path = f"{directory}/library_embeddings.npy"

    index = faiss.read_index(index_path)
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    embeddings = np.load(embeddings_path)
    assert len(records) == embeddings.shape[0], "❌ Index / JSON mismatch"

    return index, records


def search_library(
    query: str,
    k: int,
    *,
    index: faiss.Index,
    records: list[dict],
    client: OpenAI
) -> List[Dict[str, Any]]:

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_vec = np.array(
        emb.data[0].embedding,
        dtype="float32"
    ).reshape(1, -1)

    k = max(1, min(k, index.ntotal))
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        record = records[idx]
        materials = record.get("materials") or []
        material_name = materials[0].get("name") if materials else None

        results.append({
            "title": record.get("title"),
            "author": record.get("author"),
            "material": material_name,
            "year": record.get("publicationDate"),
            "summary": record.get("summary"),
            "subjects": record.get("subjects"),
            "contributors": record.get("contributors"),
            "distance": float(dist)
        })

    return results


# ---------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------

def create_conversation(client: OpenAI) -> str:
    conv = client.conversations.create()
    return conv.id


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def run_conversation_loop():
    client = get_client()
    index, records = load_library()
    conv_id = create_conversation(client)

    web_tools = get_web_tools()

    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        # -------------------------------------------------------------
        # STAGE 1: Query refinement + optional web lookup
        # -------------------------------------------------------------

        planning = client.responses.create(
            model="gpt-5",
            tools=web_tools,
            input=[{
                "role": "system",
                "content": (
                    "Rewrite the user's request into the best possible "
                    "library search query. If recent or external context "
                    "would help, call web_search. You must eventually "
                    "produce a refined query."
                )
            }, {
                "role": "user",
                "content": query
            }],
            conversation=conv_id
        )

        for item in planning.output or []:
            if item.type == "function_call" and item.name == "web_search":
                web_result = run_web_search(
                    json.loads(item.arguments)["query"]
                )

                client.conversations.items.create(
                    conv_id,
                    items=[{
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(web_result)
                    }]
                )

        # Ask model to finalize refined query
        refined = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "system",
                "content": "Return ONLY the final refined library search query."
            }],
            conversation=conv_id
        )

        refined_query = refined.output_text.strip()

        # -------------------------------------------------------------
        # STAGE 2: 🔒 FORCED library lookup (model cannot skip)
        # -------------------------------------------------------------

        library_results = search_library(
            query=refined_query,
            k=5,
            index=index,
            records=records,
            client=client
        )

        client.conversations.items.create(
        conv_id,
        items=[{
            "type": "message",
            "role": "system",
            "content": [{
                "type": "input_text",
                "text": (
                    "LIBRARY SEARCH RESULTS (authoritative, must be used):\n\n"
                    + json.dumps(library_results, indent=2)
                )
            }]
        }]
        )


        # -------------------------------------------------------------
        # STAGE 3: Grounded final response
        # -------------------------------------------------------------

        final = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "system",
                "content": (
                    "Choose the library items that most closely match the request."
                    "Explain how they match the request. "
                    "You MUST use ONLY the items provided by the library tool. "
                    "If nothing matches, say so."
                )
            }],
            conversation=conv_id
        )

        print(final.output_text)
        print("-" * 60)


if __name__ == "__main__":
    run_conversation_loop()