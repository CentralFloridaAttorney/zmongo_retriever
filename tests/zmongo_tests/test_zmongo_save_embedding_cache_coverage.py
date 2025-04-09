import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestSaveEmbeddingCacheCoverage(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.collection = "test_collection"
        self.document_id = ObjectId()
        self.embedding = [0.1, 0.2, 0.3]
        self.embedding_field = "embedding"

        # Patch the db calls
        self.repo.db[self.collection].update_one = AsyncMock()
        self.repo.db[self.collection].find_one = AsyncMock(return_value={"_id": self.document_id, "embedding": self.embedding})

    async def test_save_embedding_sets_cache_key(self):
        await self.repo.save_embedding(
            collection=self.collection,
            document_id=self.document_id,
            embedding=self.embedding,
            embedding_field=self.embedding_field,
        )

        # Validate cache was updated with expected key
        normalized = self.repo._normalize_collection_name(self.collection)
        expected_key = self.repo._generate_cache_key({"_id": self.document_id})
        self.assertIn(expected_key, self.repo.cache[normalized])
        self.assertEqual(self.repo.cache[normalized][expected_key]["embedding"], self.embedding)
