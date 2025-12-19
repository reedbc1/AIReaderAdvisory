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
from difflib import SequenceMatcher

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
# Vector helpers
# ---------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def choose_weights(query: str) -> tuple[float, float]:
    q = query.lower()

    if "like" in q and len(q.split()) <= 5:
        return 0.8, 0.2

    if any(w in q for w in ["but", "with", "without", "more", "less", "scarier", "slower", "faster"]):
        return 0.6, 0.4

    return 0.7, 0.3


def fuzzy_title_match(
    query: str,
    records: List[Dict[str, Any]],
    threshold: float = 0.85
) -> int | None:
    q = query.lower()
    best_score = 0.0
    best_idx = None

    for i, r in enumerate(records):
        title = r.get("title")
        if not title:
            continue

        score = SequenceMatcher(None, q, title.lower()).ratio()
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx if best_score >= threshold else None


# ---------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------

def load_library() -> Tuple[faiss.Index, List[Dict[str, Any]], np.ndarray]:
    directory = prompt_for_subdirectory()

    index = faiss.read_index(f"{directory}/library.index")

    with open(f"{directory}/wr_enhanced.json", encoding="utf-8") as f:
        records = json.load(f)

    embeddings = np.load(f"{directory}/library_embeddings.npy")
    assert len(records) == embeddings.shape[0], "❌ Index / JSON mismatch"

    return index, records, embeddings


# ---------------------------------------------------------------------
# FAISS search with hybrid query vector
# ---------------------------------------------------------------------

def search_library(
    query: str,
    *,
    index: faiss.Index,
    records: List[Dict[str, Any]],
    embeddings: np.ndarray,
    client: OpenAI,
    k: int = 20
) -> tuple[List[Dict[str, Any]], bool]:

    # Always embed query text
    text_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    text_vec = normalize(
        np.array(text_emb.data[0].embedding, dtype="float32")
    )

    match_idx = fuzzy_title_match(query, records)
    item_weight, text_weight = choose_weights(query)

    if match_idx is not None:
        item_vec = normalize(embeddings[match_idx])
        query_vec = normalize(
            item_weight * item_vec + text_weight * text_vec
        ).reshape(1, -1)
        exact_item_found = True
    else:
        query_vec = text_vec.reshape(1, -1)
        exact_item_found = False

    k = max(1, min(k, index.ntotal))
    distances, indices = index.search(query_vec, k)

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

    return results, exact_item_found


# ---------------------------------------------------------------------
# Python-side ranking
# ---------------------------------------------------------------------

def rank_results(
    results: List[Dict[str, Any]],
    top_n: int = 4
) -> List[Dict[str, Any]]:

    ranked = sorted(results, key=lambda r: r["distance"])

    for r in ranked:
        r["score"] = (
            (1 / (1 + r["distance"])) +
            (0.1 if r.get("summary") else 0)
        )

    ranked = sorted(ranked, key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------
# LLM explanation
# ---------------------------------------------------------------------

def explain_results(
    *,
    client: OpenAI,
    patron_query: str,
    results: List[Dict[str, Any]],
    exact_item_found: bool
) -> str:

    warning = ""
    if not exact_item_found:
        warning = (
            "⚠️ **Exact item not found in the catalog.** "
            "Recommendations are based on similar themes and descriptions.\n\n"
        )

    response = client.responses.create(
        model="gpt-4o-mini",
        max_output_tokens=300,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a professional librarian providing concise "
                    "reader's advisory recommendations.\n\n"
                    "Rules:\n"
                    "- Use bullet points\n"
                    "- Recommend at most 4 items\n"
                    "- No more than 2 sentences per item\n"
                    "- Do NOT invent books\n"
                    "- Base explanations only on the provided data"
                )
            },
            {
                "role": "user",
                "content": (
                    warning +
                    f"Patron request:\n{patron_query}\n\n"
                    "Recommended items:\n"
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
    index, records, embeddings = load_library()

    print("\n📚 Reader's Advisory Agent (fast mode)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Patron request: ").strip()
        if query.lower() == "exit":
            break

        print("\n🔍 Searching catalog...")
        raw_results, exact_item_found = search_library(
            query=query,
            index=index,
            records=records,
            embeddings=embeddings,
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
            results=top_results,
            exact_item_found=exact_item_found
        )

        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    run_conversation_loop()
