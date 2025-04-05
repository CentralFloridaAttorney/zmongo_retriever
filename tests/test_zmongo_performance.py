import time
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoPerformance(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()
        self.repo.cache.clear()
        self.collection = "perf_collection"
        self.query = {"_id": ObjectId()}
        self.normalized = self.repo._normalize_collection_name(self.collection)
        self.serialized_doc = {"_id": str(self.query["_id"]), "name": "perf_test"}

    async def asyncSetUp(self):
        self.repo.db[self.collection].find_one = AsyncMock(return_value=self.query)
        self.repo.serialize_document = MagicMock(return_value=self.serialized_doc)

    async def test_find_document_cache_performance(self):
        start = time.perf_counter()
        await self.repo.find_document(self.collection, self.query)
        cold_duration = time.perf_counter() - start

        start = time.perf_counter()
        await self.repo.find_document(self.collection, self.query)
        warm_duration = time.perf_counter() - start

        print(f"\nCold cache lookup: {cold_duration:.6f}s")
        print(f"Warm cache lookup: {warm_duration:.6f}s")

        self.assertLess(warm_duration, cold_duration * 0.5)

    async def test_bulk_write_throughput_1M(self):
        bulk_size = 1000000
        mock_op = MagicMock()
        bulk_ops = [mock_op] * bulk_size
        self.repo.db[self.collection].bulk_write = AsyncMock()

        start = time.perf_counter()
        await self.repo.bulk_write(self.collection, bulk_ops)
        duration = time.perf_counter() - start

        print(f"\nBulk write of {bulk_size:,} ops: {duration:.6f}s")
        self.repo.db[self.collection].bulk_write.assert_awaited_once()

    async def test_insert_document_batch_latency(self):
        self.repo.db[self.collection].insert_one = AsyncMock()
        batch_size = 1000
        inserted_id = ObjectId()
        self.repo.db[self.collection].insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))

        start = time.perf_counter()
        for i in range(batch_size):
            await self.repo.insert_document(self.collection, {"name": f"user_{i}"})
        duration = time.perf_counter() - start

        print(f"\nInsert {batch_size} documents: {duration:.6f}s (avg {duration/batch_size:.6f}s per insert)")

    async def test_cache_throughput_under_load(self):
        self.repo.db[self.collection].find_one = AsyncMock(return_value=self.query)

        start = time.perf_counter()
        for _ in range(10000):
            await self.repo.find_document(self.collection, self.query)
        duration = time.perf_counter() - start

        print(f"\n10,000 cached lookups: {duration:.6f}s (avg {duration / 10000:.8f}s per lookup)")


if __name__ == "__main__":
    unittest.main()
