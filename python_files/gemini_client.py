"""Shared Gemini client helpers for chat and embeddings."""

from __future__ import annotations

import os
from functools import lru_cache

import google.generativeai as genai
from dotenv import load_dotenv

EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-1.5-flash"


@lru_cache(maxsize=1)
def configure_genai() -> None:
    """Load environment variables and configure the Gemini SDK once."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)


def embed_text(text: str) -> list[float]:
    """Embed a single string using the Gemini embedding model."""
    configure_genai()
    response = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return response["embedding"]


@lru_cache(maxsize=1)
def get_chat_model(system_instruction: str | None = None) -> genai.GenerativeModel:
    """Return a cached GenerativeModel configured for reader's advisory."""
    configure_genai()
    if system_instruction is None:
        return genai.GenerativeModel(model_name=CHAT_MODEL)

    return genai.GenerativeModel(
        model_name=CHAT_MODEL,
        system_instruction=system_instruction,
    )

