"""Core recommendation logic for the Reader's Advisory service.

This module exposes a single `recommend(query: str) -> str` function while
keeping the existing FAISS + OpenAI workflow intact. It is intentionally
stateless and loads the FAISS index/catalog exactly once per process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from python_files.choose_dir import list_subdirectories, prompt_for_subdirectory

# ---------------------------------------------------------------------------
# OpenAI client (lazy, cached)
# ---------------------------------------------------------------------------


def get_client() -> OpenAI:
    """Return a cached OpenAI client, loading environment variables once."""
    if not hasattr(get_client, "_client"):
        load_dotenv()
        get_client._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[attr-defined]
    return get_client._client  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dataset loading and caching
# ---------------------------------------------------------------------------

DEFAULT_BASE_DATA_DIR = "data"
DATASET_ENV_VARS = ("RECOMMENDER_DATA_DIR", "DATASET_DIR")

_library_cache: Tuple[faiss.Index, List[Dict[str, Any]], np.ndarray] | None = None
_library_path: str | None = None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def _resolve_dataset_dir(
    directory: str | None,
    *,
    allow_prompt: bool,
    base_path: str = DEFAULT_BASE_DATA_DIR,
) -> str:
    """Resolve which dataset directory to use without prompting in servers."""
    if directory:
        return directory

    for env_var in DATASET_ENV_VARS:
        env_dir = os.getenv(env_var)
        if env_dir:
            return env_dir

    if allow_prompt:
        return prompt_for_subdirectory(base_path)

    subdirs = list_subdirectories(base_path)
    if not subdirs:
        raise FileNotFoundError(
            f"No dataset folders found in {base_path}. Set one of "
            f"{', '.join(DATASET_ENV_VARS)} or create a subfolder."
        )

    # Deterministic default for non-interactive environments
    return os.path.join(base_path, sorted(subdirs)[0])


def get_library(
    directory: str | None = None,
    *,
    allow_prompt: bool = False,
) -> Tuple[faiss.Index, List[Dict[str, Any]], np.ndarray]:
    """Load (or reuse) the FAISS index, catalog metadata, and embeddings."""
    global _library_cache, _library_path

    resolved_dir = _resolve_dataset_dir(directory, allow_prompt=allow_prompt)
    if _library_cache is not None and _library_path == resolved_dir:
        return _library_cache

    index_path = Path(resolved_dir) / "library.index"
    catalog_path = Path(resolved_dir) / "wr_enhanced.json"
    embeddings_path = Path(resolved_dir) / "library_embeddings.npy"

    if not index_path.exists() or not catalog_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing required data files in {resolved_dir}. Expected "
            f"'library.index', 'wr_enhanced.json', and 'library_embeddings.npy'."
        )

    index = faiss.read_index(str(index_path))

    with catalog_path.open(encoding="utf-8") as f:
        records = json.load(f)

    embeddings = np.load(str(embeddings_path))
    assert len(records) == embeddings.shape[0], "❌ Index / JSON mismatch"

    _library_cache = (index, records, embeddings)
    _library_path = resolved_dir
    return _library_cache


# ---------------------------------------------------------------------------
# Recommendation workflow (preserves original logic)
# ---------------------------------------------------------------------------


def search_library(
    query: str,
    *,
    index: faiss.Index,
    records: List[Dict[str, Any]],
    client: OpenAI,
    k: int = 50,
) -> List[Dict[str, Any]]:
    """Embed the query and retrieve the closest catalog items via FAISS."""
    emb = client.embeddings.create(model="text-embedding-3-small", input=query)

    query_vec = _normalize(np.array(emb.data[0].embedding, dtype="float32")).reshape(1, -1)
    distances, indices = index.search(query_vec, min(k, index.ntotal))

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        record = records[idx]
        materials = record.get("materials") or []
        results.append(
            {
                "title": record.get("title"),
                "author": record.get("author"),
                "material": materials[0].get("name") if materials else None,
                "year": record.get("publicationDate"),
                "summary": record.get("summary"),
                "subjects": record.get("subjects"),
                "contributors": record.get("contributors"),
                "distance": float(dist),
            }
        )

    return results


def prefilter_results(results: List[Dict[str, Any]], max_items: int = 15) -> List[Dict[str, Any]]:
    """Score and trim candidate items before LLM synthesis."""
    ranked = sorted(results, key=lambda r: r["distance"])

    for r in ranked:
        r["score"] = (1 / (1 + r["distance"])) + (0.1 if r.get("summary") else 0)

    ranked = sorted(ranked, key=lambda r: r["score"], reverse=True)
    return ranked[:max_items]


def explain_results(
    *,
    client: OpenAI,
    patron_query: str,
    candidates: List[Dict[str, Any]],
) -> str:
    """Ask GPT to summarize the selected candidates for the patron."""
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
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patron request:\n{patron_query}\n\n"
                    "Candidate items from the library catalog:\n"
                    + json.dumps(candidates, indent=2)
                ),
            },
        ],
    )

    return response.output_text.strip()


def recommend(query: str, *, directory: str | None = None) -> str:
    """Public entrypoint: perform a single recommendation request."""
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("Query must not be empty.")

    client = get_client()
    index, records, _ = get_library(directory, allow_prompt=False)

    raw_results = search_library(query=cleaned, index=index, records=records, client=client)
    if not raw_results:
        return "❌ No results found."

    candidates = prefilter_results(raw_results, max_items=15)
    return explain_results(client=client, patron_query=cleaned, candidates=candidates)
