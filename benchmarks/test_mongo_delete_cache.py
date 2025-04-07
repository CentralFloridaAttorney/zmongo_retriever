import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId

from zmongo_toolbag.zmongo import ZMongo


class TestZMongoDeleteCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()

    async def test_delete_document_and_cache_key_cleared(self):
        collection = "test_collection"
        query = {"_id": ObjectId()}
        normalized = self.repo._normalize_collection_name(collection)
        expected_cache_key = self.repo._generate_cache_key(query)

        # Prepopulate the cache manually
        self.repo.cache[normalized][expected_cache_key] = {"cached": True}
        self.assertIn(expected_cache_key, self.repo.cache[normalized])  # sanity check

        # Mock delete_one to return a deleted_count of 1
        mock_delete_result = MagicMock(deleted_count=1)
        self.repo.db[collection].delete_one = AsyncMock(return_value=mock_delete_result)

        await self.repo.delete_document(collection, query)

        # Assert that the specific cache key was cleared
        self.assertNotIn(expected_cache_key, self.repo.cache[normalized])
