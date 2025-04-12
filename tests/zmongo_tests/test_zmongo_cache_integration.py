import unittest
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo

class TestZMongoCacheIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.collection = "test_find_document"
        self.test_doc = {"_id": ObjectId(), "name": "test_doc"}
        await self.repo.delete_document(self.collection, {"_id": self.test_doc["_id"]})
        await self.repo.insert_document(self.collection, self.test_doc)

    async def asyncTearDown(self):
        await self.repo.delete_document(self.collection, {"_id": self.test_doc["_id"]})
        await self.repo.clear_cache()
        await self.repo.close()

    async def test_find_document_cache_miss_and_hit(self):
        # First fetch - should hit MongoDB and store in cache
        result = await self.repo.find_document(self.collection, {"_id": self.test_doc["_id"]})
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "test_doc")

        # Second fetch - should hit cache
        result_cached = await self.repo.find_document(self.collection, {"_id": self.test_doc["_id"]})
        self.assertEqual(result_cached, result)

        # Confirm cache actually contains it
        normalized = self.repo._normalize_collection_name(self.collection)
        cache_key = self.repo._generate_cache_key({"_id": str(self.test_doc["_id"])})
        self.assertIn(cache_key, self.repo.cache[normalized])
