import os
import asyncio
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Import the two systems to be benchmarked
# Assumes these files are in the same directory or accessible in the path
from zmongo import ZMongo
from zmongo_service import ZMongoService

# --- Configuration ---
load_dotenv(Path.home() / "resources" / ".env_local")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_DOCS = 100  # Use a smaller number for service tests to manage API costs/time
COLLECTION_NAME = "benchmark_collection"


# --- Data Generation ---
def generate_docs(n: int) -> List[Dict[str, Any]]:
    """Generates a list of sample documents for testing."""
    return [
        {
            "title": f"Document Title {i}",
            "content": f"This is the main text content for document number {i}. It contains unique information to test embedding and search.",
            "author": f"Author_{i % 10}"
        }
        for i in range(n)
    ]


# --- Benchmark Runner ---
async def run_benchmarks():
    """Main function to orchestrate and run the benchmarks."""
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DATABASE_NAME")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([mongo_uri, db_name, gemini_api_key]):
        print("Skipping benchmarks. Please set MONGO_URI, MONGO_DATABASE_NAME, and GEMINI_API_KEY.")
        return

    docs_to_test = generate_docs(NUM_DOCS)

    # --- ZMongo (Low-Level) Benchmark ---
    print("\n" + "=" * 50)
    print("--- Benchmarking ZMongo (Low-Level DB Throughput) ---")
    print("=" * 50)

    zm_instance = ZMongo()
    await zm_instance.db.drop_collection(COLLECTION_NAME)

    # Bulk Insert
    start_time = time.perf_counter()
    await zm_instance.insert_documents(COLLECTION_NAME, docs_to_test)
    end_time = time.perf_counter()
    insert_duration = end_time - start_time
    insert_ops_sec = NUM_DOCS / insert_duration if insert_duration > 0 else float('inf')
    print(f"Bulk Insert ({NUM_DOCS} docs): {insert_ops_sec:,.0f} ops/sec")

    # Per-Doc Find
    doc_ids_to_find = [doc["_id"] for doc in (await zm_instance.find_documents(COLLECTION_NAME, {})).data]
    start_time = time.perf_counter()
    for doc_id in doc_ids_to_find:
        await zm_instance.find_document(COLLECTION_NAME, {"_id": doc_id})
    end_time = time.perf_counter()
    find_duration = end_time - start_time
    find_ops_sec = NUM_DOCS / find_duration if find_duration > 0 else float('inf')
    print(f"Per-Doc Find ({NUM_DOCS} docs):  {find_ops_sec:,.0f} ops/sec")

    await zm_instance.close()

    # --- ZMongoService (High-Level) Benchmark ---
    print("\n" + "=" * 50)
    print("--- Benchmarking ZMongoService (High-Level End-to-End Latency) ---")
    print("=" * 50)

    service_instance = ZMongoService(
        mongo_uri=mongo_uri,
        db_name=db_name,
        gemini_api_key=gemini_api_key
    )
    await service_instance.repository.db.drop_collection(COLLECTION_NAME)

    # Add & Embed
    total_add_embed_time = 0
    for doc in docs_to_test:
        start_time = time.perf_counter()
        res = await service_instance.add_and_embed(COLLECTION_NAME, doc, "content")
        end_time = time.perf_counter()
        if not res.success:
            print(f"Failed to add/embed document: {res.error}")
            continue
        total_add_embed_time += (end_time - start_time)

    avg_add_embed_ms = (total_add_embed_time / NUM_DOCS) * 1000 if NUM_DOCS > 0 else 0
    print(f"Add & Embed ({NUM_DOCS} docs):   {avg_add_embed_ms:.2f} ms/op (average)")

    # Semantic Search
    random_query_doc = random.choice(docs_to_test)
    query_text = f"Tell me about {random_query_doc['title']}"

    start_time = time.perf_counter()
    await service_instance.search(COLLECTION_NAME, query_text)
    end_time = time.perf_counter()
    search_duration_ms = (end_time - start_time) * 1000
    print(f"Semantic Search (1 query): {search_duration_ms:.2f} ms/op")

    await service_instance.close_connection()


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
