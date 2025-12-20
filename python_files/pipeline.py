"""Unified entrypoint for Vega scraping, embedding creation, and chat."""

import argparse
import asyncio
from pathlib import Path

from catalog import (
    DEFAULT_LOCATION_IDS,
    DEFAULT_MATERIAL_TYPE_IDS,
    DEFAULT_PARTITIONS,
    enrich_snapshot,
    fetch_catalog_snapshot,
)
from conversation import run_conversation_loop
from embeddings import embed_library


async def run_async_steps(
    *,
    fetch: bool,
    embed: bool,
    run_dir: Path,
    partitions: list[str],
    location_ids: int | None,
    material_type_ids: int | None,
) -> None:
    if fetch:
        summary = await fetch_catalog_snapshot(
            run_dir=run_dir,
            partitions=partitions,
            location_ids=location_ids,
            material_type_ids=material_type_ids,
        )
        print(f"Catalog snapshot → {summary}")

        enhanced_path = await enrich_snapshot(run_dir)
        print(f"Enriched records saved to {enhanced_path}")

    if embed:
        await embed_library(str(run_dir))


def main():
    parser = argparse.ArgumentParser(description="Run Vega pipeline and chat assistant")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from Vega and enrich editions")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings and FAISS index")
    parser.add_argument("--chat", action="store_true", help="Start the interactive chat loop")
    parser.add_argument(
        "--run-dir",
        default="data/wr_run",
        help="Directory to store catalog artifacts (wr.json, wr_enhanced.json, index files)",
    )
    parser.add_argument(
        "--partitions",
        nargs="*",
        help="Custom partition keys (default: A* through Z* and 0*)",
    )
    parser.add_argument(
        "--location-id",
        type=int,
        default=DEFAULT_LOCATION_IDS,
        help="Vega locationIds filter",
    )
    parser.add_argument(
        "--material-type-id",
        type=int,
        default=DEFAULT_MATERIAL_TYPE_IDS,
        help="Vega materialTypeIds filter",
    )
    parser.add_argument(
        "--chat-dir",
        help="Directory containing wr_enhanced.json and library.index for chat. Defaults to run-dir.",
    )
    args = parser.parse_args()

    partitions = args.partitions or DEFAULT_PARTITIONS
    run_dir = Path(args.run_dir)

    if args.fetch or args.embed:
        asyncio.run(
            run_async_steps(
                fetch=args.fetch,
                embed=args.embed,
                run_dir=run_dir,
                partitions=partitions,
                location_ids=args.location_id,
                material_type_ids=args.material_type_id,
            )
        )

    if args.chat:
        run_conversation_loop(args.chat_dir or args.run_dir)


if __name__ == "__main__":
    main()
