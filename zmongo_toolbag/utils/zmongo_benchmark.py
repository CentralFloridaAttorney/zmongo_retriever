import asyncio
import time
import random
from typing import List, Dict, Any, Callable, Awaitable
from bson.objectid import ObjectId
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient

# Assumes zmongo.py is in the same package or accessible path
from zmongo_toolbag.zmongo import ZMongo


def generate_docs(n=1000, collection_name="benchmark_coll") -> List[Dict[str, Any]]:
    base_text = "The quick brown fox jumps over the lazy dog."
    return [
        {
            "_id": ObjectId(),
            "text": f"{base_text} #{i}"
        }
        for i in range(n)
    ]


class BenchmarkSystem:
    name: str

    async def insert_documents(self, collection: str, docs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    async def find_document(self, collection: str, doc_id: ObjectId) -> Dict[str, Any]:
        raise NotImplementedError

    async def update_document(self, collection: str, doc_id: ObjectId, update: dict) -> None:
        raise NotImplementedError

    async def delete_documents(self, collection: str, query: dict) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        pass  # Default no-op for closing connections


class PyMongoSystem(BenchmarkSystem):
    def __init__(self):
        self.name = "PyMongo"
        self.client = pymongo.MongoClient()
        self.db = self.client["zmongo_benchmark_test_db"]

    async def insert_documents(self, collection, docs):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.db[collection].insert_many, docs)

    async def find_document(self, collection, doc_id):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.db[collection].find_one, {"_id": doc_id})

    async def update_document(self, collection, doc_id, update):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.db[collection].update_one, {"_id": doc_id}, {"$set": update})

    async def delete_documents(self, collection, query):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.db[collection].delete_many, query)

    async def close(self):
        self.client.close()


class ZMongoSystem(BenchmarkSystem):
    def __init__(self):
        self.name = "ZMongo"
        # Let ZMongo manage its own connection via context manager
        self.zm: ZMongo

    # FIX: Simplified insert_documents to match the current ZMongo implementation
    async def insert_documents(self, collection, docs):
        await self.zm.insert_documents(collection, docs)

    async def find_document(self, collection, doc_id):
        res = await self.zm.find_document(collection, {"_id": doc_id})
        return res.data

    async def update_document(self, collection, doc_id, update):
        await self.zm.update_document(collection, {"_id": doc_id}, {"$set": update})

    async def delete_documents(self, collection, query):
        await self.zm.delete_documents(collection, query)

    # Context management for setup/teardown
    async def __aenter__(self):
        self.zm = ZMongo()
        await self.zm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.zm.__aexit__(exc_type, exc_val, exc_tb)


class MotorSystem(BenchmarkSystem):
    def __init__(self):
        self.name = "Motor"
        self.client = AsyncIOMotorClient()
        self.db = self.client["zmongo_benchmark_test_db"]

    async def insert_documents(self, collection, docs):
        await self.db[collection].insert_many(docs)

    async def find_document(self, collection, doc_id):
        return await self.db[collection].find_one({"_id": doc_id})

    async def update_document(self, collection, doc_id, update):
        await self.db[collection].update_one({"_id": doc_id}, {"$set": update})

    async def delete_documents(self, collection, query):
        await self.db[collection].delete_many(query)

    async def close(self):
        self.client.close()


class BenchmarkTask:
    def __init__(self, name: str,
                 fn: Callable[[BenchmarkSystem, str, List[Dict[str, Any]], List[ObjectId]], Awaitable[float]]):
        self.name = name
        self.fn = fn


async def task_insert(system: BenchmarkSystem, collection: str, docs: List[Dict], doc_ids: List[ObjectId]):
    await system.delete_documents(collection, {})
    t0 = time.perf_counter()
    await system.insert_documents(collection, docs)
    return time.perf_counter() - t0


async def task_find(system: BenchmarkSystem, collection: str, docs: List[Dict], doc_ids: List[ObjectId]):
    ids_to_find = random.sample(doc_ids, min(100, len(doc_ids)))
    t0 = time.perf_counter()
    for doc_id in ids_to_find:
        await system.find_document(collection, doc_id)
    return time.perf_counter() - t0


async def task_update(system: BenchmarkSystem, collection: str, docs: List[Dict], doc_ids: List[ObjectId]):
    ids_to_update = random.sample(doc_ids, min(50, len(doc_ids)))
    t0 = time.perf_counter()
    for doc_id in ids_to_update:
        await system.update_document(collection, doc_id, {"updated": True})
    return time.perf_counter() - t0


async def task_delete(system: BenchmarkSystem, collection: str, docs: List[Dict], doc_ids: List[ObjectId]):
    t0 = time.perf_counter()
    await system.delete_documents(collection, {})
    return time.perf_counter() - t0


TASKS = [
    BenchmarkTask("Insert 1000 docs", task_insert),
    BenchmarkTask("Find 100 docs", task_find),
    BenchmarkTask("Update 50 docs", task_update),
    BenchmarkTask("Delete all docs", task_delete),
]


async def run_benchmarks():
    # FIX: Removed the outdated ZMongo modes
    system_classes = [ZMongoSystem, MotorSystem, PyMongoSystem]
    results = {}

    print("\nBenchmarking systems side by side:")
    for task in TASKS:
        print(f"\n--- {task.name} ---")
        collection = f"benchmark_{task.name.replace(' ', '_').lower()}_{random.getrandbits(32):x}"
        docs = generate_docs(1000)
        doc_ids = [doc["_id"] for doc in docs]

        # Pre-populate data for find and update tasks
        if "Find" in task.name or "Update" in task.name:
            # Use a separate client for setup to not interfere with benchmarked client
            setup_client = MotorSystem()
            await setup_client.delete_documents(collection, {})
            await setup_client.insert_documents(collection, docs)
            await setup_client.close()

        task_results = {}
        for SysClass in system_classes:
            system = SysClass()

            # Use async with for ZMongoSystem to handle setup/teardown
            if isinstance(system, ZMongoSystem):
                async with system:
                    t = await task.fn(system, collection, docs, doc_ids)
            else:
                t = await task.fn(system, collection, docs, doc_ids)
                await system.close()

            task_results[system.name] = t
            print(f"{system.name:<10} : {t:.4f} seconds")
        results[task.name] = task_results

    print("\n--- Summary Table ---")
    system_names = [s().name for s in system_classes]
    header = f"| {'Task':<20} | " + " | ".join(f"{name:<10}" for name in system_names) + " |"
    separator = "|-" + "-" * 20 + "-|-" + "-|-".join(["-" * 10] * len(system_names)) + "-|"
    print(header)
    print(separator)

    for task_name, task_data in results.items():
        row = f"| {task_name:<20} | "
        for name in system_names:
            row += f"{task_data.get(name, float('nan')):<10.4f} | "
        print(row)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())