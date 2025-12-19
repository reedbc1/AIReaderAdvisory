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
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from choose_dir import prompt_for_subdirectory
from gemini_client import embed_text, get_chat_model


# ---------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


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
# FAISS search (PURE semantic similarity)
# ---------------------------------------------------------------------

def search_library(
    query: str,
    *,
    index: faiss.Index,
    records: List[Dict[str, Any]],
    k: int = 50
) -> List[Dict[str, Any]]:

    # Embed the raw user query only
    emb = embed_text(query)
    query_vec = normalize(np.array(emb, dtype="float32")).reshape(1, -1)

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

def extract_gemini_text(response) -> str:
    """
    Safely extract text from a Gemini generate_content response.
    Handles safety-blocked or empty responses without throwing.
    """
    if getattr(response, "candidates", None):
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if not content or not getattr(content, "parts", None):
                continue

            texts = [
                part.text
                for part in content.parts
                if hasattr(part, "text") and part.text
            ]

            if texts:
                return "\n".join(texts).strip()

    # Optional: inspect safety ratings during debugging
    # for c in response.candidates or []:
    #     print("Safety ratings:", getattr(c, "safety_ratings", None))

    return (
        "⚠️ I couldn’t generate a reader’s advisory for this request.\n"
        "This may be due to content restrictions or weak matches.\n"
        "You might try rephrasing or exploring a related theme."
    )


def explain_results(
    *,
    patron_query: str,
    candidates: List[Dict[str, Any]]
) -> str:

    system_prompt = (
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

    model = get_chat_model(system_prompt)

    response = model.generate_content(
        [
            {
                "role": "user",
                "parts": [
                    (
                        f"Patron request:\n{patron_query}\n\n"
                        "Candidate items from the library catalog:\n"
                        + json.dumps(candidates, indent=2)
                    )
                ],
            }
        ],
        generation_config={"max_output_tokens": 350},
    )

    return extract_gemini_text(response)



# ---------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------

def run_conversation_loop() -> None:
    index, records, _ = load_library()

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
            records=records
        )

        if not raw_results:
            print("\n❌ No results found.\n")
            continue

        candidates = prefilter_results(raw_results, max_items=15)

        print("📖 Reviewing results...\n")
        answer = explain_results(
            patron_query=query,
            candidates=candidates
        )

        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    run_conversation_loop()
