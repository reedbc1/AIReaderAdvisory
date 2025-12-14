"""Unified entrypoint for Vega scraping, embedding creation, and chat."""

import argparse
import asyncio

from conversation import run_conversation_loop
from embeddings import embed_library
from stateful_pipeline import sync_catalog_state


async def run_async_steps(fetch, embed):
    if fetch:
        summary = await sync_catalog_state()
        print(f"Catalog sync → {summary}")

    if embed:
        await embed_library()


def main():
    parser = argparse.ArgumentParser(description="Run Vega pipeline and chat assistant")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from Vega and enrich editions")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings and FAISS index")
    parser.add_argument("--chat", action="store_true", help="Start the interactive chat loop")
    args = parser.parse_args()

    if args.fetch or args.embed:
        asyncio.run(run_async_steps(args.fetch, args.embed))

    if args.chat:
        run_conversation_loop()


if __name__ == "__main__":
    main()
