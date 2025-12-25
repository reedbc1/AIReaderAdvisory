"""
Fast, stateless interactive conversation loop for the
WR (Weber Road) Reader's Advisory recommendation agent.

Design goals:
- No conversational memory
- No model-based ranking
- Minimal tokens
- Fast response time
"""

from __future__ import annotations

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
# OpenAI client
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------

def load_library() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
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


# ---------------------------------------------------------------------
# FAISS search
# ---------------------------------------------------------------------

def search_library(
    query: str,
    *,
    index: faiss.Index,
    records: List[Dict[str, Any]],
    client: OpenAI,
    k: int = 20
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

    results: List[Dict[str, Any]] = []

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
            "distance": float(dist),
        })

    return results


# ---------------------------------------------------------------------
# Python-side ranking (CRITICAL FOR SPEED)
# ---------------------------------------------------------------------

def rank_results(
    results: List[Dict[str, Any]],
    top_n: int = 20
) -> List[Dict[str, Any]]:

    # Base ranking: FAISS distance
    ranked = sorted(results, key=lambda r: r["distance"])

    # Lightweight heuristic scoring
    for r in ranked:
        r["score"] = (
            (1 / (1 + r["distance"])) +
            (0.1 if r.get("summary") else 0)
        )

    ranked = sorted(ranked, key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------
# LLM explanation (small, fast)
# ---------------------------------------------------------------------

def explain_results(
    *,
    client: OpenAI,
    patron_query: str,
    results: List[Dict[str, Any]]
) -> str:

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a professional librarian providing concise "
                    "reader's advisory recommendations.\n\n"
                    "Rules:\n"
                    "- Use bullet points\n"
                    "- Recommend at most 10 items\n"
                    "- No more than 2 sentences per item\n"
                    "- Do NOT invent items\n"
                    "- Base explanations only on the provided data"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Patron request:\n{patron_query}\n\n"
                    "Selected library items:\n"
                    + json.dumps(results, indent=2)
                )
            }
        ]
    )

    return response.output_text.strip()


# ---------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------

def run_conversation_loop() -> None:
    client = get_client()
    index, records = load_library()

    print("\n📚 Reader's Advisory Agent (fast mode)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Patron request: ").strip()
        if query.lower() == "exit":
            break

        print("\n🔍 Searching catalog...")
        raw_results = search_library(
            query=query,
            index=index,
            records=records,
            client=client,
            k=20
        )

        if not raw_results:
            print("\n❌ No results found.\n")
            continue

        top_results = rank_results(raw_results, top_n=4)

        print("📖 Preparing recommendations...\n")
        answer = explain_results(
            client=client,
            patron_query=query,
            results=top_results
        )

        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    run_conversation_loop()
