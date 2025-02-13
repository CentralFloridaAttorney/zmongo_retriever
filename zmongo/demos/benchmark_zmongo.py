#!/usr/bin/env python3
"""
benchmark_zmongo.py

A simple script that benchmarks ZMongoRepository under high load. By default,
it inserts a large number of documents concurrently to test throughput.

Usage (from terminal):
    python benchmark_zmongo.py --collection test_bench --num_docs 5000 --concurrency 10

Or simply run without arguments to use the default settings:
    python benchmark_zmongo.py
"""

import argparse
import asyncio
import math
import random
import string
import time

from zmongo.BAK.zmongo_repository import ZMongoRepository


async def insert_worker(
    repo: ZMongoRepository,
    collection: str,
    num_docs: int,
    worker_id: int,
):
    """
    A single worker that inserts 'num_docs' documents into 'collection'.
    Each document is randomly generated.
    """
    for i in range(num_docs):
        # Generate a pseudo-random document
        doc = {
            "test_worker_id": worker_id,
            "doc_index": i,
            "random_text": "".join(random.choices(string.ascii_letters, k=10)),
            "random_number": random.randint(0, 10_000),
            "random_float": random.random(),
        }
        await repo.insert_document(collection, doc)


async def run_inserts(
    repo: ZMongoRepository,
    collection: str,
    total_docs: int,
    concurrency: int,
):
    """
    Create 'concurrency' tasks that collectively insert 'total_docs' documents.
    Distributes docs among workers as evenly as possible.
    """
    tasks = []
    docs_per_worker = math.ceil(total_docs / concurrency)
    for worker_id in range(concurrency):
        # The last worker might not need to do docs_per_worker if total_docs
        # is not perfectly divisible
        start_doc_index = worker_id * docs_per_worker
        end_doc_index = min((worker_id + 1) * docs_per_worker, total_docs)
        num_docs_for_worker = end_doc_index - start_doc_index

        if num_docs_for_worker > 0:
            tasks.append(
                insert_worker(repo, collection, num_docs_for_worker, worker_id)
            )

    await asyncio.gather(*tasks)


async def benchmark_inserts(args):
    """
    Benchmark insert performance by creating a large number of documents.
    """
    # Instantiate the repository
    repo = ZMongoRepository()

    # Warm up the connection (optional)
    # You can do a trivial operation to ensure the initial connection overhead
    # is not counted in the main test
    await repo.find_document(args.collection, {})

    # Start timing
    start_time = time.perf_counter()

    # Perform concurrent inserts
    await run_inserts(repo, args.collection, args.num_docs, args.concurrency)

    # Calculate total time and throughput
    end_time = time.perf_counter()
    total_time = end_time - start_time
    docs_per_sec = args.num_docs / total_time if total_time > 0 else 0

    print(f"Insertion Benchmark Results:")
    print(f"  Total documents inserted: {args.num_docs}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Total time (seconds): {total_time:.2f}")
    print(f"  Throughput (docs/sec): {docs_per_sec:.2f}")

    # Close the repository connection
    await repo.close()


async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark ZMongoRepository under load.")
    parser.add_argument(
        "--collection",
        type=str,
        default="test_bench",
        help="MongoDB collection name to use for benchmarking.",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=5000,
        help="Number of documents to insert for benchmarking.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent workers to insert documents.",
    )
    args = parser.parse_args()

    # Run the insert benchmark
    await benchmark_inserts(args)


if __name__ == "__main__":
    asyncio.run(main())
