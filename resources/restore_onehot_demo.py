#!/usr/bin/env python3
"""
restore_words.py

Restores and prints the original words from a list of one-hot indices
stored in OneHotDB.

Usage:
  python restore_words.py my_doc_key \
      --collection sentences \
      --vocab-collection onehot_vocab
"""

from __future__ import annotations

import argparse
import asyncio
import sys

try:
    from zmongo_toolbag.zmongo import ZMongo
    from zmongo_toolbag.zonehotdb import ZOneHotDB
except ImportError:
    print("Error: Could not import ZMongo or OneHotDB.", file=sys.stderr)
    print("Please ensure zonehotdb.py and zmongo.py are in the correct path.", file=sys.stderr)
    sys.exit(1)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Restore words from one-hot indices in the database.")
    parser.add_argument("link_key", type=str, help="The link_key of the sentence to restore.")
    parser.add_argument("--collection", type=str, default="sentences",
                        help="Mongo collection where encoded sentences are stored (default: sentences)")
    parser.add_argument("--vocab-collection", type=str, default="onehot_vocab",
                        help="Mongo collection where the vocabulary is stored (default: onehot_vocab)")

    args = parser.parse_args()

    async with ZMongo() as repo:
        # Initialize OneHotDB, pointing to the correct collections
        oh = ZOneHotDB(
            _table_name=args.collection,
            repo=repo,
            vocab_collection=args.vocab_collection
        )

        # 1. Get the list of index strings (e.g., ['1', '54', '23'])
        index_list = await oh.get_onehot_list(args.link_key)

        if not index_list:
            print(f"Error: No indices found for link_key '{args.link_key}' in collection '{args.collection}'.", file=sys.stderr)
            return 1

        print(f"Found {len(index_list)} indices for link_key '{args.link_key}'. Restoring words...")
        print("-" * 30)

        # 2. Asynchronously look up each word from its index
        restored_words = []
        # Create a list of tasks to run concurrently
        tasks = [oh.lex.get_word(int(idx)) for idx in index_list]
        # Run all lookup tasks
        results = await asyncio.gather(*tasks)

        for word in results:
            if word:
                restored_words.append(word)
            else:
                restored_words.append("[WORD NOT FOUND]") # Handle case where an index is invalid

        # 3. Print the restored sentence
        restored_sentence = " ".join(restored_words)
        print(restored_sentence)
        print("-" * 30)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
