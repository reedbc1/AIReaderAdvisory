"""FastAPI entrypoint for the Reader's Advisory service."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.recommender import get_library, recommend

app = FastAPI(
    title="Reader's Advisory",
    description="Stateless semantic recommendation API.",
    version="1.0.0",
)


class RecommendRequest(BaseModel):
    query: str


class RecommendResponse(BaseModel):
    answer: str


@app.on_event("startup")
def load_resources() -> None:
    """Load FAISS index and catalog once at startup."""
    # Avoid interactive prompts inside server environments.
    get_library(allow_prompt=False)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_endpoint(payload: RecommendRequest) -> RecommendResponse:
    """Stateless recommendation endpoint."""
    try:
        answer = await run_in_threadpool(recommend, payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surface unexpected errors clearly
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RecommendResponse(answer=answer)


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    """Serve the minimal frontend if available."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Ensure static/index.html is present.",
        )

    return HTMLResponse(index_path.read_text(encoding="utf-8"))
