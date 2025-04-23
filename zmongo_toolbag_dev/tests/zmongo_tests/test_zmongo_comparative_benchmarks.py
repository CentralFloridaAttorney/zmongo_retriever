import asyncio
import time
import threading
import unittest
from bson import ObjectId
from pymongo import MongoClient, InsertOne
import redis
from unittest.mock import AsyncMock, MagicMock

from zmongo_toolbag.zmongo import ZMongo


class TestZMongoAndRealDBComparativeBenchmarks(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Setup real Mongo and Redis connections
        cls.mongo_client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        cls.db = cls.mongo_client["ztarot"]
        cls.mongo_collection = cls.db["benchmark"]

        cls.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        cls.redis_client.set("benchmark_key", "value")  # preload for cache hit testing

        # Setup ZMongo mock environment
        cls.zmongo = ZMongo()
        cls.collection = "benchmark_collection"
        cls.query = {"_id": ObjectId()}
        cls.serialized_doc = {"_id": str(cls.query["_id"]), "name": "benchmark"}
        cls.normalized = cls.zmongo._normalize_collection_name(cls.collection)

    @classmethod
    def tearDownClass(cls):
        cls.mongo_client.close()
        cls.redis_client.close()

    async def asyncSetUp(self):
        self.zmongo.db = MagicMock()
        self.zmongo.cache.clear()
        self.zmongo.db[self.collection].find_one = AsyncMock(return_value=self.query)
        self.zmongo.serialize_document = MagicMock(return_value=self.serialized_doc)

    def test_bulk_write_mongo(self):
        docs = [{"_id": ObjectId(), "val": i} for i in range(100000)]
        start = time.perf_counter()
        self.mongo_collection.insert_many(docs)
        duration = time.perf_counter() - start
        ops_per_sec = len(docs) / duration
        print(f"\nMongoDB bulk insert (100k docs): {ops_per_sec:,.2f} ops/sec")

    def test_query_latency_mongo(self):
        doc_id = ObjectId()
        self.mongo_collection.insert_one({"_id": doc_id, "val": "test"})
        rounds = 5000
        start = time.perf_counter()
        for _ in range(rounds):
            _ = self.mongo_collection.find_one({"_id": doc_id})
        duration = time.perf_counter() - start
        avg_latency = duration / rounds
        print(f"\nMongoDB query latency (cached document): {avg_latency * 1000:.4f} ms")

    def test_insert_latency_mongo(self):
        start = time.perf_counter()
        for i in range(500):
            self.mongo_collection.insert_one({"name": f"user_{i}"})
        duration = time.perf_counter() - start
        avg_insert = duration / 500
        print(f"\nMongoDB insert 500 docs: {duration:.4f}s (avg {avg_insert * 1000:.4f} ms per insert)")

    def test_concurrent_reads_mongo(self):
        doc_id = ObjectId()
        self.mongo_collection.insert_one({"_id": doc_id, "val": "test"})
        count = 5000
        start = time.perf_counter()

        def read():
            self.mongo_collection.find_one({"_id": doc_id})

        threads = [threading.Thread(target=read) for _ in range(count)]
        for t in threads: t.start()
        for t in threads: t.join()
        duration = time.perf_counter() - start
        print(f"\nMongoDB 5k concurrent reads (threads): {duration:.4f}s")

    def test_redis_get_latency(self):
        rounds = 5000
        start = time.perf_counter()
        for _ in range(rounds):
            _ = self.redis_client.get("benchmark_key")
        duration = time.perf_counter() - start
        avg_latency = duration / rounds
        print(f"\nRedis GET latency (cached key): {avg_latency * 1000:.4f} ms")

    def test_redis_insert_latency(self):
        start = time.perf_counter()
        for i in range(500):
            self.redis_client.set(f"user_{i}", f"val_{i}")
        duration = time.perf_counter() - start
        avg_latency = duration / 500
        print(f"\nRedis SET 500 keys: {duration:.4f}s (avg {avg_latency * 1000:.4f} ms per insert)")

    def test_redis_concurrent_reads(self):
        key = "benchmark_key"
        count = 5000
        start = time.perf_counter()

        def read():
            self.redis_client.get(key)

        threads = [threading.Thread(target=read) for _ in range(count)]
        for t in threads: t.start()
        for t in threads: t.join()
        duration = time.perf_counter() - start
        print(f"\nRedis 5k concurrent GETs (threads): {duration:.4f}s")

    async def test_bulk_write_throughput_mocked(self):
        bulk_ops = [MagicMock() for _ in range(100000)]
        self.zmongo.db[self.collection].bulk_write = AsyncMock()
        start = time.perf_counter()
        await self.zmongo.bulk_write(self.collection, bulk_ops)
        duration = time.perf_counter() - start
        ops_per_second = len(bulk_ops) / duration
        print(f"\nBulk write 100k ops throughput (mocked): {ops_per_second:,.2f} ops/sec")

    async def test_query_latency_under_mocked_load(self):
        query_count = 5000
        start = time.perf_counter()
        for _ in range(query_count):
            await self.zmongo.find_document(self.collection, self.query)
        duration = time.perf_counter() - start
        avg_latency = duration / query_count
        print(f"\nAverage query latency (cached, mocked): {avg_latency * 1000:.4f} ms")

    async def test_cache_hit_ratio(self):
        await self.zmongo.find_document(self.collection, self.query)
        self.zmongo.db[self.collection].find_one = AsyncMock(side_effect=Exception("Should not be called"))
        hits = 0
        for _ in range(10000):
            result = await self.zmongo.find_document(self.collection, self.query)
            if result:
                hits += 1
        hit_ratio = hits / 10000
        print(f"\nCache hit ratio: {hit_ratio:.2%}")

    async def test_insert_latency_under_mocked_batch(self):
        inserted_id = ObjectId()
        self.zmongo.db[self.collection].insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))
        start = time.perf_counter()
        for i in range(500):
            await self.zmongo.insert_document(self.collection, {"name": f"test_{i}"})
        duration = time.perf_counter() - start
        avg_insert = duration / 500
        print(f"\nInsert 500 docs (mocked): {duration:.4f}s (avg {avg_insert * 1000:.4f} ms per insert)")

    async def test_simulated_concurrent_reads_mocked(self):
        async def one_read():
            await self.zmongo.find_document(self.collection, self.query)
        start = time.perf_counter()
        await asyncio.gather(*(one_read() for _ in range(5000)))
        duration = time.perf_counter() - start
        print(f"\nConcurrent read test (5000 ops, mocked): {duration:.4f}s total")


if __name__ == "__main__":
    unittest.main()

