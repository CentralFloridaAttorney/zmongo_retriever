import time
import unittest
from bson import ObjectId
from pymongo import MongoClient
import redis
import threading


class TestRealDBComparativeBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mongo_client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        cls.db = cls.mongo_client["ztarot"]
        cls.mongo_collection = cls.db["benchmark"]

        cls.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        cls.redis_client.set("benchmark_key", "value")  # preload for cache hit testing

    @classmethod
    def tearDownClass(cls):
        cls.mongo_client.close()
        cls.redis_client.close()

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
        inserted_ids = []
        start = time.perf_counter()
        for i in range(500):
            inserted_ids.append(self.mongo_collection.insert_one({"name": f"user_{i}"}).inserted_id)
        duration = time.perf_counter() - start
        avg_insert = duration / len(inserted_ids)

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


if __name__ == "__main__":
    unittest.main()
