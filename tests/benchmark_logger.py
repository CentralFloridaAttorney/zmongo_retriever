import time
import asyncio
import threading
from bson import ObjectId
from pymongo import MongoClient, InsertOne
from redis import Redis
from zmongo_toolbag.zmongo import ZMongo


class BenchmarkLogger:
    def __init__(self):
        self.results = []

    def log(self, category, system, value, unit, notes=""):
        self.results.append({
            "category": category,
            "system": system,
            "value": value,
            "unit": unit,
            "notes": notes
        })

    def save_to_txt(self, path="benchmark_results.txt"):
        with open(path, "w") as f:
            f.write("ZMongo Retriever Real-World Benchmark Comparison\n")
            f.write("=" * 60 + "\n\n")
            categories = sorted(set(r["category"] for r in self.results))
            for cat in categories:
                f.write(f"{cat}\n" + "-" * len(cat) + "\n")
                for entry in filter(lambda r: r["category"] == cat, self.results):
                    f.write(f"{entry['system']:>15}: {entry['value']:.4f} {entry['unit']}  {entry['notes']}\n")
                f.write("\n")
            f.write("=" * 60 + "\n")


logger = BenchmarkLogger()
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_db = mongo_client["ztarot"]
mongo_col = mongo_db["benchmarks_raw"]
redis_client = Redis(host="localhost", port=6379, db=0)
redis_client.set("benchmark_key", "value")
zmongo = ZMongo()
zmongo_col = "benchmarks_zmongo"


async def run_benchmarks():
    await zmongo.delete_all_documents(zmongo_col)

    from random import randint
    from typing import List

    # ...rest of your code...

    # Benchmark: insert_documents (ZMongo vs MongoDB vs Redis)
    documents: List[dict] = [{"val": randint(1, 100)} for _ in range(100_000)]

    # ZMongo insert_documents
    await zmongo.delete_all_documents(zmongo_col)
    start = time.perf_counter()
    inserted_count = await zmongo.insert_documents(zmongo_col, documents)
    duration = time.perf_counter() - start
    logger.log("insert_documents (100k)", "ZMongo", inserted_count / duration, "ops/sec")

    # MongoDB Shell equivalent
    docs_with_ids = [{"_id": ObjectId(), "val": randint(1, 100)} for _ in range(100_000)]
    mongo_col.delete_many({})
    start = time.perf_counter()
    mongo_col.insert_many(docs_with_ids)
    duration = time.perf_counter() - start
    logger.log("insert_documents (100k)", "MongoDB Shell", len(docs_with_ids) / duration, "ops/sec")

    # Redis equivalent (not 1:1, but approximated)
    start = time.perf_counter()
    for i in range(100_000):
        redis_client.set(f"bulk_user_{i}", f"value_{i}")
    duration = time.perf_counter() - start
    logger.log("insert_documents (100k)", "Redis", 100_000 / duration, "ops/sec")

    # MongoDB Shell Bulk Write
    docs = [{"_id": ObjectId(), "val": i} for i in range(100_000)]
    mongo_col.delete_many({})
    start = time.perf_counter()
    mongo_col.insert_many(docs)
    duration = time.perf_counter() - start
    logger.log("Bulk Write (100k)", "MongoDB Shell", len(docs)/duration, "ops/sec")

    # ZMongo Bulk Write
    ops = [InsertOne({"index": i}) for i in range(100_000)]
    start = time.perf_counter()
    await zmongo.bulk_write(zmongo_col, ops)
    duration = time.perf_counter() - start
    logger.log("Bulk Write (100k)", "ZMongo", len(ops)/duration, "ops/sec")

    # Insert Latency
    start = time.perf_counter()
    for i in range(500):
        mongo_col.insert_one({"val": i})
    duration = time.perf_counter() - start
    logger.log("Insert (500 docs)", "MongoDB Shell", duration / 500 * 1000, "ms/doc")

    start = time.perf_counter()
    for i in range(500):
        await zmongo.insert_document(zmongo_col, {"val": i})
    duration = time.perf_counter() - start
    logger.log("Insert (500 docs)", "ZMongo", duration / 500 * 1000, "ms/doc")

    start = time.perf_counter()
    for i in range(500):
        redis_client.set(f"user_{i}", f"value_{i}")
    duration = time.perf_counter() - start
    logger.log("Insert (500 docs)", "Redis", duration / 500 * 1000, "ms/doc")

    # Query Latency (Cached)
    doc_id = ObjectId()
    mongo_col.insert_one({"_id": doc_id, "val": "cached"})
    for _ in range(10): mongo_col.find_one({"_id": doc_id})  # warm cache
    start = time.perf_counter()
    for _ in range(5000):
        mongo_col.find_one({"_id": doc_id})
    duration = time.perf_counter() - start
    logger.log("Query Latency (cached)", "MongoDB Shell", duration / 5000 * 1000, "ms")

    cached_doc = {"_id": ObjectId(), "val": "cached"}
    await zmongo.insert_document(zmongo_col, cached_doc)
    await zmongo.find_document(zmongo_col, {"_id": cached_doc["_id"]})  # warm cache
    start = time.perf_counter()
    for _ in range(5000):
        await zmongo.find_document(zmongo_col, {"_id": cached_doc["_id"]})
    duration = time.perf_counter() - start
    logger.log("Query Latency (cached)", "ZMongo", duration / 5000 * 1000, "ms")

    start = time.perf_counter()
    for _ in range(5000):
        redis_client.get("benchmark_key")
    duration = time.perf_counter() - start
    logger.log("Query Latency (cached)", "Redis", duration / 5000 * 1000, "ms")

    # Concurrent Reads
    doc_id = ObjectId()
    mongo_col.insert_one({"_id": doc_id, "val": "concurrent"})
    threads = [threading.Thread(target=lambda: mongo_col.find_one({"_id": doc_id})) for _ in range(5000)]
    start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    duration = time.perf_counter() - start
    logger.log("Concurrent Reads (5k)", "MongoDB Shell", duration, "s")

    redis_threads = [threading.Thread(target=lambda: redis_client.get("benchmark_key")) for _ in range(5000)]
    start = time.perf_counter()
    for t in redis_threads: t.start()
    for t in redis_threads: t.join()
    duration = time.perf_counter() - start
    logger.log("Concurrent Reads (5k)", "Redis", duration, "s")

    async def one_read():
        await zmongo.find_document(zmongo_col, {"_id": cached_doc["_id"]})

    start = time.perf_counter()
    await asyncio.gather(*(one_read() for _ in range(5000)))
    duration = time.perf_counter() - start
    logger.log("Concurrent Reads (5k)", "ZMongo", duration, "s")

    logger.save_to_txt("benchmark_results.txt")

import asyncio

# your existing async def run_benchmarks() and other code...

async def main():
    await run_benchmarks()
    print("✅ Benchmark completed and saved to benchmark_results.txt")

if __name__ == "__main__":
    asyncio.run(main())
