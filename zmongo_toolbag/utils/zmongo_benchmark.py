import asyncio
import time
import random
from typing import List, Dict, Any, Callable, Awaitable
from bson import ObjectId

def generate_docs(n=1000, collection_name="benchmark_coll") -> List[Dict[str, Any]]:
    base_text = "The quick brown fox jumps over the lazy dog."
    return [
        {
            "_id": ObjectId(),
            "database_name": "testdb",
            "collection_name": collection_name,
            "casebody": {
                "data": {
                    "opinions": [{"text": f"{base_text} #{i}"}]
                }
            }
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
    async def delete_documents(self, collection: str) -> None:
        raise NotImplementedError

import pymongo

class PyMongoSystem(BenchmarkSystem):
    def __init__(self):
        self.name = "PyMongo"
        self.client = pymongo.MongoClient()
        self.db = self.client["test"]
    async def insert_documents(self, collection, docs):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.db[collection].insert_many, docs)
    async def find_document(self, collection, doc_id):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.db[collection].find_one, {"_id": doc_id})
    async def update_document(self, collection, doc_id, update):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.db[collection].update_one, {"_id": doc_id}, {"$set": update})
    async def delete_documents(self, collection):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.db[collection].delete_many, {})

from zmongo_toolbag.zmongo import ZMongo
class ZMongoSystem(BenchmarkSystem):
    def __init__(self, mode="normal"):
        self.name = "ZMongo" if mode=="normal" else f"ZMongo ({mode})"
        self.zm = ZMongo()
        self.mode = mode

    async def insert_documents(self, collection, docs):
        if self.mode == "fast":
            await self.zm.insert_documents(collection, docs, fast_mode=True)
        elif self.mode == "buffered":
            try:
                await self.zm.insert_documents(collection, docs, buffer_only=True)
                await self.zm.flush_buffered_inserts(collection)
            except Exception as e:
                print(f"ERROR: Buffered insert failed for {self.name} on {collection}: {e}")
                raise
        else:
            await self.zm.insert_documents(collection, docs)

    async def find_document(self, collection, doc_id):
        return (await self.zm.find_document(collection, {"_id": doc_id})).data
    async def update_document(self, collection, doc_id, update):
        await self.zm.update_document(collection, {"_id": doc_id}, update)
    async def delete_documents(self, collection):
        await self.zm.delete_documents(collection)

from motor.motor_asyncio import AsyncIOMotorClient
class MotorSystem(BenchmarkSystem):
    def __init__(self):
        self.name = "Motor"
        self.client = AsyncIOMotorClient()
        self.db = self.client["test"]
    async def insert_documents(self, collection, docs):
        await self.db[collection].insert_many(docs)
    async def find_document(self, collection, doc_id):
        return await self.db[collection].find_one({"_id": doc_id})
    async def update_document(self, collection, doc_id, update):
        await self.db[collection].update_one({"_id": doc_id}, {"$set": update})
    async def delete_documents(self, collection):
        await self.db[collection].delete_many({})

class BenchmarkTask:
    def __init__(self, name: str, fn: Callable[[BenchmarkSystem, str, List[Dict[str, Any]], List[ObjectId]], Awaitable[float]]):
        self.name = name
        self.fn = fn

async def task_insert(system: BenchmarkSystem, collection: str, docs, doc_ids):
    await system.delete_documents(collection)
    t0 = time.perf_counter()
    try:
        await system.insert_documents(collection, docs)
    except Exception as e:
        print(f"Insert failed for {system.name} in {collection}: {e}")
        return float('nan')
    t1 = time.perf_counter()
    return t1 - t0

async def task_find(system: BenchmarkSystem, collection: str, docs, doc_ids):
    ids = random.sample(doc_ids, min(100, len(doc_ids)))
    t0 = time.perf_counter()
    try:
        for _id in ids:
            _ = await system.find_document(collection, _id)
    except Exception as e:
        print(f"Find failed for {system.name} in {collection}: {e}")
        return float('nan')
    t1 = time.perf_counter()
    return t1 - t0

async def task_update(system: BenchmarkSystem, collection: str, docs, doc_ids):
    ids = random.sample(doc_ids, min(50, len(doc_ids)))
    t0 = time.perf_counter()
    try:
        for _id in ids:
            await system.update_document(collection, _id, {"updated": True})
    except Exception as e:
        print(f"Update failed for {system.name} in {collection}: {e}")
        return float('nan')
    t1 = time.perf_counter()
    return t1 - t0

async def task_delete(system: BenchmarkSystem, collection: str, docs, doc_ids):
    t0 = time.perf_counter()
    try:
        await system.delete_documents(collection)
    except Exception as e:
        print(f"Delete failed for {system.name} in {collection}: {e}")
        return float('nan')
    t1 = time.perf_counter()
    return t1 - t0

TASKS = [
    BenchmarkTask("Insert 1000 docs", task_insert),
    BenchmarkTask("Find 100 docs", task_find),
    BenchmarkTask("Update 50 docs", task_update),
    BenchmarkTask("Delete all docs", task_delete),
]

async def run_benchmarks():
    systems = [
        ZMongoSystem(),  # normal
        ZMongoSystem(mode="fast"),
        ZMongoSystem(mode="buffered"),
        MotorSystem(),
        PyMongoSystem(),
    ]
    results = {}
    print("\nBenchmarking systems side by side:")
    for task in TASKS:
        print(f"\n--- {task.name} ---")
        results[task.name] = {}
        collection = f"benchmark_coll_{random.getrandbits(48):x}"
        docs = generate_docs(1000, collection_name=collection)
        doc_ids = [_doc["_id"] for _doc in docs]
        for sys in systems:
            try:
                await sys.delete_documents(collection)
            except Exception as e:
                print(f"Delete pre-task failed for {sys.name}: {e}")
        # For insert: no pre-insert needed, for others: pre-insert for each system
        if task.name != "Insert 1000 docs":
            for sys in systems:
                sys_docs = generate_docs(1000, collection_name=collection)
                sys_doc_ids = [_doc["_id"] for _doc in sys_docs]
                try:
                    await sys.insert_documents(collection, sys_docs)
                except Exception as e:
                    print(f"Pre-task insert failed for {sys.name}: {e}")
                if sys == systems[0]:
                    docs = sys_docs
                    doc_ids = sys_doc_ids
        for sys in systems:
            try:
                t = await task.fn(sys, collection, docs, doc_ids)
            except Exception as e:
                print(f"ERROR in {task.name} for {sys.name}: {e}")
                t = float('nan')
            results[task.name][sys.name] = t
            print(f"{sys.name:<20} : {t:.4f} seconds")
    print("\n--- Summary Table ---")
    systems_names = [sys.name for sys in systems]
    header = "| Task | " + " | ".join(systems_names) + " |"
    sep = "|---" * (len(systems_names) + 1) + "|"
    print(header)
    print(sep)
    for task in TASKS:
        row = f"| {task.name} | " + " | ".join(f"{results[task.name][sys]:.4f}" for sys in systems_names) + " |"
        print(row)

if __name__ == "__main__":
    asyncio.run(run_benchmarks())
