import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoUpdateDocument(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.zmongo = ZMongo()
        self.collection = "update_test"
        self.query = {"_id": ObjectId()}
        self.update_data = {"$set": {"name": "Test"}}
        self.updated_doc = {"_id": self.query["_id"], "name": "Test"}

    async def test_update_document_success_and_exception(self):
        # ✅ Mock successful update_one + find_one call
        mock_collection_success = MagicMock()
        mock_result = MagicMock(matched_count=1, modified_count=1, upserted_id=None)
        mock_collection_success.update_one = AsyncMock(return_value=mock_result)
        mock_collection_success.find_one = AsyncMock(return_value=self.updated_doc)

        # ✅ Inject into self.zmongo.db[collection]
        self.zmongo.db = MagicMock()
        self.zmongo.db.__getitem__.return_value = mock_collection_success

        result = await self.zmongo.update_document(self.collection, self.query, self.update_data)
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(self.zmongo.cache[self.collection][self.zmongo._generate_cache_key(self.query)]["name"], "Test")

        # ✅ Mock update_one to raise exception
        mock_collection_fail = MagicMock()
        mock_collection_fail.update_one = AsyncMock(side_effect=RuntimeError("update failed"))
        self.zmongo.db.__getitem__.return_value = mock_collection_fail

        with self.assertRaises(RuntimeError) as exc_info:
            await self.zmongo.update_document(self.collection, self.query, self.update_data)

        self.assertIn("update failed", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
