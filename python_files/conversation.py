"""Interactive FAISS search loop for library records."""

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
    client: OpenAI,
) -> List[Dict[str, Any]]:
    """Embed the query, search FAISS, and return the top matches."""

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_vec = np.array(
        embedding.data[0].embedding,
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


def format_result(result: Dict[str, Any], rank: int) -> str:
    """Pretty-print a single search result."""

    lines = [f"{rank}. {result.get('title') or 'Untitled'}"]

    author = result.get("author")
    if author:
        lines.append(f"   Author: {author}")

    material = result.get("material")
    if material:
        lines.append(f"   Material: {material}")

    year = result.get("year")
    if year:
        lines.append(f"   Publication Date: {year}")

    contributors = result.get("contributors")
    if contributors:
        lines.append(f"   Contributors: {contributors}")

    subjects = result.get("subjects")
    if subjects:
        lines.append(f"   Subjects: {subjects}")

    summary = result.get("summary")
    if summary:
        lines.append(f"   Summary: {summary}")

    lines.append(f"   Distance: {result.get('distance'):.4f}")
    return "\n".join(lines)


def run_search_loop():
    """Prompt for queries and print the 100 closest matches from FAISS."""

    client = get_client()
    index, records = load_library()

    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("Please enter a search query.\n")
            continue

        print("Searching library...\n")

        results = search_library(
            query=query,
            k=100,
            index=index,
            records=records,
            client=client,
        )

        if not results:
            print("No results found.\n")
            continue

        for idx, result in enumerate(results, start=1):
            print(format_result(result, idx))
            print()


if __name__ == "__main__":
    run_search_loop()
