import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()

    async def test_cache_key_generation_and_normalization(self):
        query = {"_id": "123"}
        key = self.repo._generate_cache_key(query)
        self.assertIsInstance(key, str)

        collection = "Test_Collection"
        normalized = self.repo._normalize_collection_name(collection)
        self.assertEqual(normalized, "test_collection")

    async def test_find_document_cache_miss_and_store(self):
        collection = "test_collection"
        query = {"_id": ObjectId()}

        mock_doc = {"_id": query["_id"], "value": 42}
        self.repo.db[collection].find_one = AsyncMock(return_value=mock_doc)

        result = await self.repo.find_document(collection, query)

        normalized = self.repo._normalize_collection_name(collection)
        cache_key = self.repo._generate_cache_key(query)
        self.assertEqual(result["value"], 42)
        self.assertIn(cache_key, self.repo.cache[normalized])

    async def test_find_document_cache_hit(self):
        collection = "test_collection"
        query = {"_id": ObjectId()}

        normalized = self.repo._normalize_collection_name(collection)
        cache_key = self.repo._generate_cache_key(query)
        self.repo.cache[normalized][cache_key] = {"_id": str(query["_id"]), "value": "cached"}

        result = await self.repo.find_document(collection, query)
        self.assertEqual(result["value"], "cached")

    async def test_find_document_cache_miss_no_document(self):
        collection = "test_collection"
        query = {"_id": ObjectId()}

        self.repo.db[collection].find_one = AsyncMock(return_value=None)
        result = await self.repo.find_document(collection, query)
        self.assertIsNone(result)


    async def test_insert_document_stores_in_cache(self):
        collection = "test_collection"
        document = {"name": "John"}
        inserted_id = ObjectId()

        # Mock result
        mock_result = MagicMock()
        mock_result.inserted_id = inserted_id
        self.repo.db[collection].insert_one = AsyncMock(return_value=mock_result)

        # Call insert_document
        result = await self.repo.insert_document(collection, document)

        # Since the insert_document method now returns InsertOneResult,
        # we need to extract inserted_id from it.
        self.assertIsInstance(result, MagicMock)  # result should be the InsertOneResult
        self.assertEqual(result.inserted_id, inserted_id)  # Ensure inserted_id is correct

        # Normalize collection and generate cache key
        normalized = self.repo._normalize_collection_name(collection)
        cache_key = self.repo._generate_cache_key({"_id": str(inserted_id)})

        # Check that the cache contains the serialized version
        self.assertIn(cache_key, self.repo.cache[normalized])
        self.assertEqual(self.repo.cache[normalized][cache_key]["_id"]["$oid"], str(inserted_id))

    async def test_delete_document_clears_cache_key(self):
        collection = "test_collection"
        query = {"_id": ObjectId()}
        normalized = self.repo._normalize_collection_name(collection)
        cache_key = self.repo._generate_cache_key(query)

        self.repo.cache[normalized][cache_key] = {"cached": True}
        mock_result = MagicMock(deleted_count=1)
        self.repo.db[collection].delete_one = AsyncMock(return_value=mock_result)

        await self.repo.delete_document(collection, query)
        self.assertNotIn(cache_key, self.repo.cache[normalized])

    async def test_delete_document_noop_if_not_found(self):
        collection = "test_collection"
        query = {"_id": ObjectId()}

        self.repo.db[collection].delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))
        await self.repo.delete_document(collection, query)  # Should not throw

    async def test_clear_cache(self):
        self.repo.cache["some_collection"]["some_key"] = {"cached": True}
        await self.repo.clear_cache()
        self.assertEqual(len(self.repo.cache), 0)


if __name__ == "__main__":
    unittest.main()
