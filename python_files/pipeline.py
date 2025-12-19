"""Unified entrypoint for Vega scraping, embedding creation, and FAISS search."""

import argparse
import asyncio

from conversation import run_search_loop
from embeddings import embed_library
from stateful_pipeline import sync_catalog_state


async def run_async_steps(fetch, embed):
    if fetch:
        summary = await sync_catalog_state()
        print(f"Catalog sync → {summary}")

    if embed:
        await embed_library()


def main():
    parser = argparse.ArgumentParser(description="Run Vega pipeline and FAISS search")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from Vega and enrich editions")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings and FAISS index")
    parser.add_argument("--search", action="store_true", help="Start the interactive FAISS search loop")
    args = parser.parse_args()

    if args.fetch or args.embed:
        asyncio.run(run_async_steps(args.fetch, args.embed))

    if args.search:
        run_search_loop()


if __name__ == "__main__":
    main()
