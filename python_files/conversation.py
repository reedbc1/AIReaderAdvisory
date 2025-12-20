"""
Fast, stateless interactive conversation loop for the
WR (Weber Road) Reader's Advisory recommendation agent.

Design goals:
- No conversational memory
- FAISS controls retrieval
- Pure semantic similarity (no lexical anchoring)
- GPT synthesizes from candidate results
- Minimal tokens
- Fast response time
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------
# Local data selection
# ---------------------------------------------------------------------

def list_subdirectories(directory: str) -> list[str]:
    base = Path(directory)
    if not base.is_dir():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def prompt_for_subdirectory(base_path: str = "data") -> str:
    subdirs = list_subdirectories(base_path)
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in: {base_path}")

    if len(subdirs) == 1:
        return str(Path(base_path) / subdirs[0])

    prompt_lines = ["Choose a folder in data/:"]
    prompt_lines.extend(
        [f"  {idx + 1}. {name}" for idx, name in enumerate(subdirs)])
    prompt_lines.append("Enter number (default 1): ")
    prompt_text = "\n".join(prompt_lines)

    while True:
        choice = input(prompt_text).strip()
        if not choice:
            return str(Path(base_path) / subdirs[0])
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(subdirs):
                return str(Path(base_path) / subdirs[idx])
        print("Invalid selection. Please try again.\n")


# ---------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


# ---------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------

def load_library(data_dir: str | None = None) -> Tuple[faiss.Index, List[Dict[str, Any]], np.ndarray]:
    directory = data_dir or prompt_for_subdirectory()

    index = faiss.read_index(f"{directory}/library.index")

    with open(f"{directory}/wr_enhanced.json", encoding="utf-8") as f:
        records = json.load(f)

    embeddings = np.load(f"{directory}/library_embeddings.npy")
    assert len(records) == embeddings.shape[0], "❌ Index / JSON mismatch"

    return index, records, embeddings


# ---------------------------------------------------------------------
# FAISS search (PURE semantic similarity)
# ---------------------------------------------------------------------

def search_library(
    query: str,
    *,
    index: faiss.Index,
    records: List[Dict[str, Any]],
    client: OpenAI,
    k: int = 50
) -> List[Dict[str, Any]]:

    # Embed the raw user query only
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_vec = normalize(
        np.array(emb.data[0].embedding, dtype="float32")
    ).reshape(1, -1)

    distances, indices = index.search(query_vec, min(k, index.ntotal))

    results: List[Dict[str, Any]] = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        r = records[idx]
        materials = r.get("materials") or []

        results.append({
            "title": r.get("title"),
            "author": r.get("author"),
            "material": materials[0].get("name") if materials else None,
            "year": r.get("publicationDate"),
            "summary": r.get("summary"),
            "subjects": r.get("subjects"),
            "contributors": r.get("contributors"),
            "distance": float(dist),
        })

    return results


# ---------------------------------------------------------------------
# Python-side pre-filtering (NOT final selection)
# ---------------------------------------------------------------------

def prefilter_results(
    results: List[Dict[str, Any]],
    max_items: int = 15
) -> List[Dict[str, Any]]:

    ranked = sorted(results, key=lambda r: r["distance"])

    for r in ranked:
        r["score"] = (
            (1 / (1 + r["distance"])) +
            (0.1 if r.get("summary") else 0)
        )

    ranked = sorted(ranked, key=lambda r: r["score"], reverse=True)
    return ranked[:max_items]


# ---------------------------------------------------------------------
# LLM synthesis (GPT reads FAISS results)
# ---------------------------------------------------------------------

def explain_results(
    *,
    client: OpenAI,
    patron_query: str,
    candidates: List[Dict[str, Any]]
) -> str:

    response = client.responses.create(
        model="gpt-4o",
        max_output_tokens=350,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a professional librarian providing concise reader's advisory.\n\n"
                    "Rules:\n"
                    "- Review the candidate items provided\n"
                    "- Select the best matches for the patron request\n"
                    "- Recommend at most 5 items\n"
                    "- Use bullet points\n"
                    "- No more than 2 sentences per item\n"
                    "- Use ONLY the provided items\n"
                    "- Do NOT invent books\n"
                    "- Prefer thematic and tonal similarity over exact word matches\n"
                    "- If none fit well, say so"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Patron request:\n{patron_query}\n\n"
                    "Candidate items from the library catalog:\n"
                    + json.dumps(candidates, indent=2)
                )
            }
        ]
    )

    return response.output_text.strip()


# ---------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------

def run_conversation_loop(data_dir: str | None = None) -> None:
    client = get_client()
    index, records, _ = load_library(data_dir)

    print("\n📚 Reader's Advisory Agent (pure semantic similarity)")
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
            client=client
        )

        if not raw_results:
            print("\n❌ No results found.\n")
            continue

        candidates = prefilter_results(raw_results, max_items=15)

        print("📖 Reviewing results...\n")
        answer = explain_results(
            client=client,
            patron_query=query,
            candidates=candidates
        )

        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    run_conversation_loop()
