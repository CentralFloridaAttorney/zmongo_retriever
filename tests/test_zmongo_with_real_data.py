import time
import asyncio
import unittest
from bson import ObjectId
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from pymongo import InsertOne


class TestZMongoWithRealData(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.collection = "benchmark_real"
        await self.repo.delete_all_documents(self.collection)

    async def test_real_bulk_write(self):
        ops = [InsertOne({"index": i}) for i in range(100_000)]
        start = time.perf_counter()
        await self.repo.bulk_write(self.collection, ops)
        duration = time.perf_counter() - start

        ops_per_sec = len(ops) / duration
        print(f"\nZMongo real bulk write (100k): {ops_per_sec:,.2f} ops/sec")

    async def test_real_insert_latency(self):
        total = 500
        start = time.perf_counter()
        for i in range(total):
            await self.repo.insert_document(self.collection, {"index": i})
        duration = time.perf_counter() - start

        avg = duration / total
        print(f"\nZMongo insert 500 docs: {duration:.4f}s (avg {avg * 1000:.4f} ms per insert)")

    async def test_real_query_latency_cached(self):
        doc = {"_id": ObjectId(), "value": "cached"}
        await self.repo.insert_document(self.collection, doc)

        # First access populates cache
        await self.repo.find_document(self.collection, {"_id": doc["_id"]})

        total = 5000
        start = time.perf_counter()
        for _ in range(total):
            await self.repo.find_document(self.collection, {"_id": doc["_id"]})
        duration = time.perf_counter() - start
        avg = duration / total

        print(f"\nZMongo cached query latency: {avg * 1000:.4f} ms/query")

    async def test_concurrent_reads(self):
        doc = {"_id": ObjectId(), "value": "test"}
        await self.repo.insert_document(self.collection, doc)

        await self.repo.find_document(self.collection, {"_id": doc["_id"]})

        async def one_read():
            await self.repo.find_document(self.collection, {"_id": doc["_id"]})

        start = time.perf_counter()
        await asyncio.gather(*(one_read() for _ in range(5000)))
        duration = time.perf_counter() - start

        print(f"\nZMongo concurrent reads (5k async): {duration:.4f}s")

if __name__ == "__main__":
    asyncio.run(unittest.main())
