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
# Library loading + search
# ---------------------------------------------------------------------

# return index, records
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

    # creates embeddings for query
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    # reshape query array
    query_vec = np.array(
        emb.data[0].embedding,
        dtype="float32"
    ).reshape(1, -1)

    # ensure k is an appropriate value
    k = max(1, min(k, index.ntotal))

    # perform search on index
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

    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        # -------------------------------------------------------------
        # STAGE 1: Query refinement (NO web search)
        # -------------------------------------------------------------

        print("Analyzing query...")

        planning = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "system",
                "content": (
                    "Rewrite the user's request into the best possible "
                    "library catalog search query. "
                    "Do NOT add outside knowledge. "
                    "Return a refined query."
                )
            }, {
                "role": "user",
                "content": query
            }],
            conversation=conv_id
        )

        refined_query = planning.output_text.strip()

        print("refined_query")

        # -------------------------------------------------------------
        # STAGE 2: 🔒 FORCED library lookup
        # -------------------------------------------------------------

        print("Searching library...")
        
        library_results = search_library(
            query=refined_query,
            k=20,
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
        # STAGE 3: Grounded final response / follow-up
        # -------------------------------------------------------------

        print("Analyzing results...")
        
        final = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "system",
                "content": (
                    "Choose the library items that most closely match the request. "
                    "Explain how they match. "
                    "You MUST use ONLY the provided library results. "
                    "If nothing matches, say so. "
                    "If clarification is needed, ask a follow-up question."
                )
            }],
            conversation=conv_id
        )

        print(final.output_text)
        print("-" * 60)


if __name__ == "__main__":
    run_conversation_loop()
