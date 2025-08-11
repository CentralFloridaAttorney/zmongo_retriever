import asyncio
import time
import random
from typing import List, Dict, Any, Callable, Awaitable
from bson.objectid import ObjectId
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient

# Assumes zmongo.py is in the same package or accessible path
# Make sure to install the dependency: pip install zmongo-toolbag
from zmongo_toolbag.zmongo import ZMongo

# --- Configuration ---
# Set the number of documents for the benchmark to match the old test
NUM_DOCS = 10000


def generate_docs(n: int) -> List[Dict[str, Any]]:
    """Generates a list of sample documents for testing."""
    base_text = "The quick brown fox jumps over the lazy dog."
    return [
        {
            "_id": ObjectId(),
            "text": f"{base_text} #{i}",
            "index": i
        }
        for i in range(n)
    ]


# --- System Abstractions ---
# These classes provide a consistent interface for each database library

class BenchmarkSystem:
    """Abstract base class for a system to be benchmarked."""
    name: str

    async def insert_many(self, collection: str, docs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    async def insert_one(self, collection: str, doc: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def find_all(self, collection: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def find_one(self, collection: str, doc_id: ObjectId) -> Dict[str, Any]:
        raise NotImplementedError

    async def delete_many(self, collection: str, query: dict) -> None:
        raise NotImplementedError

    async def delete_one(self, collection: str, doc_id: ObjectId) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass  # Default no-op for closing connections


class PyMongoSystem(BenchmarkSystem):
    """Wrapper for synchronous PyMongo to be used in an async context."""

    def __init__(self):
        self.name = "PyMongo"
        self.client = pymongo.MongoClient()
        self.db = self.client["zmongo_benchmark_test_db"]

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def insert_many(self, collection, docs):
        await self._run_sync(self.db[collection].insert_many, docs)

    async def insert_one(self, collection, doc):
        await self._run_sync(self.db[collection].insert_one, doc)

    async def find_all(self, collection):
        cursor = await self._run_sync(self.db[collection].find, {})
        return await self._run_sync(list, cursor)

    async def find_one(self, collection, doc_id):
        return await self._run_sync(self.db[collection].find_one, {"_id": doc_id})

    async def delete_many(self, collection, query):
        await self._run_sync(self.db[collection].delete_many, query)

    async def delete_one(self, collection, doc_id):
        await self._run_sync(self.db[collection].delete_one, {"_id": doc_id})

    async def close(self):
        self.client.close()


class ZMongoSystem(BenchmarkSystem):
    """Wrapper for ZMongo."""

    def __init__(self):
        self.name = "ZMongo"
        self.zm: ZMongo

    async def insert_many(self, collection, docs):
        await self.zm.insert_documents(collection, docs)

    async def insert_one(self, collection, doc):
        await self.zm.insert_document(collection, doc)

    async def find_all(self, collection):
        res = await self.zm.find_documents(collection, {})
        return res.data

    async def find_one(self, collection, doc_id):
        res = await self.zm.find_document(collection, {"_id": doc_id})
        return res.data

    async def delete_many(self, collection, query):
        await self.zm.delete_documents(collection, query)

    async def delete_one(self, collection, doc_id):
        await self.zm.delete_document(collection, {"_id": doc_id})

    async def __aenter__(self):
        self.zm = ZMongo()
        await self.zm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.zm.__aexit__(exc_type, exc_val, exc_tb)


class MotorSystem(BenchmarkSystem):
    """Wrapper for Motor."""

    def __init__(self):
        self.name = "Motor"
        self.client = AsyncIOMotorClient()
        self.db = self.client["zmongo_benchmark_test_db"]

    async def insert_many(self, collection, docs):
        await self.db[collection].insert_many(docs)

    async def insert_one(self, collection, doc):
        await self.db[collection].insert_one(doc)

    async def find_all(self, collection):
        cursor = self.db[collection].find({})
        return await cursor.to_list(length=NUM_DOCS)

    async def find_one(self, collection, doc_id):
        return await self.db[collection].find_one({"_id": doc_id})

    async def delete_many(self, collection, query):
        await self.db[collection].delete_many(query)

    async def delete_one(self, collection, doc_id):
        await self.db[collection].delete_one({"_id": doc_id})

    async def close(self):
        self.client.close()


# --- Benchmark Tasks ---
# Each function defines a specific operation to measure.

async def task_bulk_insert(system: BenchmarkSystem, collection: str, docs: List[Dict], **kwargs):
    await system.delete_many(collection, {})
    t0 = time.perf_counter()
    await system.insert_many(collection, docs)
    return time.perf_counter() - t0


async def task_per_doc_insert(system: BenchmarkSystem, collection: str, docs: List[Dict], **kwargs):
    await system.delete_many(collection, {})
    t0 = time.perf_counter()
    for doc in docs:
        await system.insert_one(collection, doc)
    return time.perf_counter() - t0


async def task_bulk_find(system: BenchmarkSystem, collection: str, **kwargs):
    t0 = time.perf_counter()
    await system.find_all(collection)
    return time.perf_counter() - t0


async def task_per_doc_find(system: BenchmarkSystem, collection: str, doc_ids: List[ObjectId], **kwargs):
    t0 = time.perf_counter()
    for doc_id in doc_ids:
        await system.find_one(collection, doc_id)
    return time.perf_counter() - t0


async def task_bulk_delete(system: BenchmarkSystem, collection: str, **kwargs):
    t0 = time.perf_counter()
    await system.delete_many(collection, {})
    return time.perf_counter() - t0


async def task_per_doc_delete(system: BenchmarkSystem, collection: str, doc_ids: List[ObjectId], **kwargs):
    t0 = time.perf_counter()
    for doc_id in doc_ids:
        await system.delete_one(collection, doc_id)
    return time.perf_counter() - t0


class BenchmarkTask:
    """Helper class to define a benchmark task."""

    def __init__(self, name: str, fn: Callable, num_ops: int):
        self.name = name
        self.fn = fn
        self.num_ops = num_ops


TASKS = [
    BenchmarkTask("Bulk Insert", task_bulk_insert, NUM_DOCS),
    BenchmarkTask("Per-Doc Insert", task_per_doc_insert, NUM_DOCS),
    BenchmarkTask("Bulk Find", task_bulk_find, NUM_DOCS),
    BenchmarkTask("Per-Doc Find", task_per_doc_find, NUM_DOCS),
    BenchmarkTask("Bulk Delete", task_bulk_delete, NUM_DOCS),
    BenchmarkTask("Per-Doc Delete", task_per_doc_delete, NUM_DOCS),
]


async def run_benchmarks():
    """Main function to orchestrate and run the benchmarks."""
    system_classes = [ZMongoSystem, MotorSystem, PyMongoSystem]
    results = {}

    print(f"--- Starting Benchmark (N={NUM_DOCS}) ---")

    # Generate docs once
    docs = generate_docs(NUM_DOCS)
    doc_ids = [doc["_id"] for doc in docs]

    for task in TASKS:
        print(f"\n--- Testing: {task.name} ---")
        collection = f"bm_{task.name.replace(' ', '_').lower()}"

        # Pre-populate data for find and delete tasks
        if "Find" in task.name or "Delete" in task.name:
            print("Pre-populating data...")
            setup_client = MotorSystem()
            await setup_client.delete_many(collection, {})
            await setup_client.insert_many(collection, docs)
            await setup_client.close()

        task_results = {}
        for SysClass in system_classes:
            system = SysClass()

            # Use async with for ZMongoSystem to handle setup/teardown
            try:
                if isinstance(system, ZMongoSystem):
                    async with system:
                        t = await task.fn(system=system, collection=collection, docs=docs, doc_ids=doc_ids)
                else:
                    t = await task.fn(system=system, collection=collection, docs=docs, doc_ids=doc_ids)
            finally:
                if not isinstance(system, ZMongoSystem):
                    await system.close()

            # Calculate throughput: operations per second
            ops_per_sec = task.num_ops / t if t > 0 else float('inf')
            task_results[system.name] = ops_per_sec
            print(f"{system.name:<10}: {ops_per_sec:,.0f} ops/sec")
        results[task.name] = task_results

    # --- Print Summary Table ---
    print("\n--- Summary Table (Throughput in Operations/Second) ---")
    system_names = [s().name for s in system_classes]
    header = f"| {'Operation':<15} | " + " | ".join(f"{name:<15}" for name in system_names) + " |"
    separator = "|-" + "-" * 15 + "-|-" + "-|-".join(["-" * 15] * len(system_names)) + "-|"
    print(header)
    print(separator)

    for task_name, task_data in results.items():
        row = f"| {task_name:<15} | "
        for name in system_names:
            ops = task_data.get(name, 0)
            row += f"{ops:<15,.0f} | "
        print(row)


if __name__ == "__main__":
    # Ensure you have the required libraries:
    # pip install pymongo motor zmongo-toolbag
    asyncio.run(run_benchmarks())
