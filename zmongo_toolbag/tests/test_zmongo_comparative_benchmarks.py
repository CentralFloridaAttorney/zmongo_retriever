import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoComparativeBenchmarks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()
        self.repo.cache.clear()
        self.collection = "benchmark_collection"
        self.query = {"_id": ObjectId()}
        self.normalized = self.repo._normalize_collection_name(self.collection)
        self.serialized_doc = {"_id": str(self.query["_id"]), "name": "benchmark"}

    async def asyncSetUp(self):
        self.repo.db[self.collection].find_one = AsyncMock(return_value=self.query)
        self.repo.serialize_document = MagicMock(return_value=self.serialized_doc)

    async def test_bulk_write_throughput_vs_baseline(self):
        bulk_ops = [MagicMock() for _ in range(100000)]
        self.repo.db[self.collection].bulk_write = AsyncMock()

        start = time.perf_counter()
        await self.repo.bulk_write(self.collection, bulk_ops)
        duration = time.perf_counter() - start

        ops_per_second = len(bulk_ops) / duration
        print(f"\nBulk write 100k ops throughput: {ops_per_second:,.2f} ops/sec")
        self.assertGreater(ops_per_second, 500_000, "Throughput lower than expected baseline (500k ops/sec)")

    async def test_query_latency_under_load(self):
        self.repo.db[self.collection].find_one = AsyncMock(return_value=self.query)
        query_count = 5000

        start = time.perf_counter()
        for _ in range(query_count):
            await self.repo.find_document(self.collection, self.query)
        duration = time.perf_counter() - start
        avg_latency = duration / query_count

        print(f"\nAverage query latency (cached): {avg_latency * 1000:.4f} ms")
        self.assertLess(avg_latency, 0.001, "Avg latency exceeds 1ms per query")

    async def test_cache_hit_ratio(self):
        # Ensure cache is populated
        await self.repo.find_document(self.collection, self.query)

        # Clear DB call to test cache-only
        self.repo.db[self.collection].find_one = AsyncMock(side_effect=Exception("Should not be called"))

        hits = 0
        total = 10000
        for _ in range(total):
            result = await self.repo.find_document(self.collection, self.query)
            if result:
                hits += 1

        hit_ratio = hits / total
        print(f"\nCache hit ratio: {hit_ratio:.2%}")
        self.assertEqual(hit_ratio, 1.0, "Cache hit ratio must be 100%")

    async def test_insert_latency_under_batch_load(self):
        inserted_id = ObjectId()
        self.repo.db[self.collection].insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))

        batch_size = 500
        start = time.perf_counter()
        for i in range(batch_size):
            await self.repo.insert_document(self.collection, {"name": f"test_{i}"})
        duration = time.perf_counter() - start
        avg_insert = duration / batch_size

        print(f"\nInsert {batch_size} docs: {duration:.4f}s (avg {avg_insert * 1000:.4f} ms per insert)")
        self.assertLess(avg_insert, 0.005, "Avg insert latency > 5ms (mocked)")

    async def test_simulated_concurrent_reads(self):
        self.repo.db[self.collection].find_one = AsyncMock(return_value=self.query)
        self.repo.serialize_document = MagicMock(return_value=self.serialized_doc)

        async def one_read():
            await self.repo.find_document(self.collection, self.query)

        start = time.perf_counter()
        await asyncio.gather(*(one_read() for _ in range(5000)))
        duration = time.perf_counter() - start

        print(f"\nConcurrent read test (5000 ops): {duration:.4f}s total")
        self.assertLess(duration, 5.0, "Concurrent reads too slow")


if __name__ == "__main__":
    unittest.main()
