import time
import asyncio
import unittest
import threading
from bson import ObjectId
from pymongo import MongoClient, InsertOne
from redis import Redis

from zmongo_toolbag.zmongo import ZMongo


class TestRealVsZMongoBenchmarks(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Raw MongoDB (blocking)
        cls.mongo_client = MongoClient("mongodb://localhost:27017")
        cls.mongo_db = cls.mongo_client["ztarot"]
        cls.mongo_col = cls.mongo_db["benchmarks_raw"]

        # Redis client
        cls.redis_client = Redis(host="localhost", port=6379, db=0)
        cls.redis_client.set("benchmark_key", "value")

    @classmethod
    def tearDownClass(cls):
        cls.mongo_client.close()

    async def asyncSetUp(self):
        self.zmongo = ZMongo()
        self.zmongo_col = "benchmarks_zmongo"
        await self.zmongo.delete_all_documents(self.zmongo_col)

    def test_mongo_shell_bulk_write(self):
        docs = [{"_id": ObjectId(), "val": i} for i in range(100_000)]
        self.mongo_col.delete_many({})
        start = time.perf_counter()
        self.mongo_col.insert_many(docs)
        duration = time.perf_counter() - start

        ops_per_sec = len(docs) / duration
        print(f"\nMongoDB Shell bulk write: {ops_per_sec:,.2f} ops/sec")

    async def test_zmongo_bulk_write(self):
        ops = [InsertOne({"index": i}) for i in range(100_000)]
        start = time.perf_counter()
        await self.zmongo.bulk_write(self.zmongo_col, ops)
        duration = time.perf_counter() - start
        print(f"ZMongo bulk write: {len(ops) / duration:,.2f} ops/sec")

    def test_mongo_shell_insert_latency(self):
        self.mongo_col.delete_many({})
        total = 500
        start = time.perf_counter()
        for i in range(total):
            self.mongo_col.insert_one({"val": i})
        duration = time.perf_counter() - start
        avg = duration / total

        print(f"\nMongoDB insert 500 docs: {duration:.4f}s (avg {avg * 1000:.4f} ms)")

    async def test_zmongo_insert_latency(self):
        total = 500
        start = time.perf_counter()
        for i in range(total):
            await self.zmongo.insert_document(self.zmongo_col, {"val": i})
        duration = time.perf_counter() - start
        avg = duration / total
        print(f"ZMongo insert 500 docs: {duration:.4f}s (avg {avg * 1000:.4f} ms)")

    def test_mongo_shell_query_latency(self):
        doc_id = ObjectId()
        self.mongo_col.insert_one({"_id": doc_id, "val": "cached"})
        total = 5000
        start = time.perf_counter()
        for _ in range(total):
            self.mongo_col.find_one({"_id": doc_id})
        duration = time.perf_counter() - start
        print(f"\nMongoDB shell query (cached): {duration/total*1000:.4f} ms/query")

    async def test_zmongo_cached_query_latency(self):
        doc = {"_id": ObjectId(), "val": "cached"}
        await self.zmongo.insert_document(self.zmongo_col, doc)
        await self.zmongo.find_document(self.zmongo_col, {"_id": doc["_id"]})  # cache warm-up

        total = 5000
        start = time.perf_counter()
        for _ in range(total):
            await self.zmongo.find_document(self.zmongo_col, {"_id": doc["_id"]})
        duration = time.perf_counter() - start
        print(f"ZMongo cached query: {duration / total * 1000:.4f} ms/query")

    def test_redis_latency(self):
        total = 5000
        start = time.perf_counter()
        for _ in range(total):
            self.redis_client.get("benchmark_key")
        duration = time.perf_counter() - start
        print(f"\nRedis GET latency: {duration / total * 1000:.4f} ms")

    def test_concurrent_reads_mongo_shell(self):
        doc_id = ObjectId()
        self.mongo_col.insert_one({"_id": doc_id, "val": "concurrent"})
        threads = [threading.Thread(target=lambda: self.mongo_col.find_one({"_id": doc_id})) for _ in range(5000)]
        start = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        duration = time.perf_counter() - start
        print(f"\nMongoDB concurrent reads (5k): {duration:.4f}s")

    async def test_concurrent_reads_zmongo(self):
        doc = {"_id": ObjectId(), "val": "concurrent"}
        await self.zmongo.insert_document(self.zmongo_col, doc)
        await self.zmongo.find_document(self.zmongo_col, {"_id": doc["_id"]})

        async def one_read():
            await self.zmongo.find_document(self.zmongo_col, {"_id": doc["_id"]})

        start = time.perf_counter()
        await asyncio.gather(*(one_read() for _ in range(5000)))
        duration = time.perf_counter() - start
        print(f"ZMongo concurrent reads (5k async): {duration:.4f}s")

    def test_redis_concurrent_reads(self):
        threads = [threading.Thread(target=lambda: self.redis_client.get("benchmark_key")) for _ in range(5000)]
        start = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        duration = time.perf_counter() - start
        print(f"Redis concurrent reads (5k): {duration:.4f}s")


if __name__ == "__main__":
    unittest.main()
