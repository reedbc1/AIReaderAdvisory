"""
Fast, stateless interactive conversation loop for the
WR (Weber Road) Reader's Advisory recommendation agent.

This script keeps the original terminal experience but delegates
all recommendation logic to `backend.recommender`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.recommender import (  # noqa: E402
    explain_results,
    get_client,
    get_library,
    prefilter_results,
    search_library,
)


def run_conversation_loop() -> None:
    client = get_client()
    index, records, _ = get_library(allow_prompt=True)

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
            client=client,
        )

        if not raw_results:
            print("\n❌ No results found.\n")
            continue

        candidates = prefilter_results(raw_results, max_items=15)

        print("📖 Reviewing results...\n")
        answer = explain_results(
            client=client,
            patron_query=query,
            candidates=candidates,
        )

        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    run_conversation_loop()
