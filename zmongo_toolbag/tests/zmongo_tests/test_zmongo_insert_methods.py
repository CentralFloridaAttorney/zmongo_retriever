import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from zmongo_toolbag.zmongo import ZMongo
from bson import ObjectId


class TestZMongoInsertMethods(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.collection = "test_collection"
        self.addCleanup(self.repo.sync_client.close)
        self.addCleanup(self.repo.mongo_client.close)

    async def asyncTearDown(self):
        if not isinstance(self.repo.db, MagicMock):
            await self.repo.delete_all_documents(self.collection)

    @patch("zmongo_toolbag.zmongo.logger")
    async def test_insert_document_handles_exception(self, mock_logger):
        collection_name = "test_collection"
        document = {"invalid": True}

        # Patch the entire collection to simulate an exception
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock(side_effect=Exception("Mocked insertion failure"))
        self.repo.db = MagicMock()
        self.repo.db.__getitem__.return_value = mock_collection

        result = await self.repo.insert_document(collection_name, document)

        self.assertIsNone(result)
        mock_logger.error.assert_called_once()
        self.assertIn("Error inserting document into", mock_logger.error.call_args[0][0])

    @patch("zmongo_toolbag.zmongo.logger")
    async def test_insert_documents_logs_error_on_failure(self, mock_logger):
        collection = "test"
        docs = [{"name": "Doc 1"}, {"name": "Doc 2"}]

        mock_collection = MagicMock()
        mock_collection.insert_many = AsyncMock(side_effect=Exception("Batch insert failed"))
        self.repo.db = MagicMock()
        self.repo.db.__getitem__.return_value = mock_collection

        result = await self.repo.insert_documents(collection, docs, batch_size=2)

        self.assertEqual(result["inserted_count"], 0)
        self.assertIn("errors", result)
        mock_logger.error.assert_called_once()
        self.assertIn("Batch insert failed", mock_logger.error.call_args[0][0])

    def test_insert_documents_sync_success(self):
        self.repo.sync_db[self.collection].delete_many({})
        docs = [{"name": f"sync_doc_{i}"} for i in range(5)]
        result = self.repo.insert_documents_sync(self.collection, docs)
        self.assertEqual(result["inserted_count"], 5)
        self.assertNotIn("errors", result)

    def test_insert_documents_sync_with_error(self):
        self.repo.sync_db[self.collection].delete_many({})
        docs = [{"_id": 1, "x": 1}, {"_id": 1, "x": 2}]  # Duplicate _id to force error
        result = self.repo.insert_documents_sync(self.collection, docs)
        self.assertLess(result["inserted_count"], 2)
        self.assertIn("errors", result)

    async def test_insert_documents_with_use_sync_true(self):
        await self.repo.delete_all_documents(self.collection)
        docs = [{"name": f"mixed_doc_{i}"} for i in range(3)]
        result = await self.repo.insert_documents(self.collection, docs, use_sync=True)
        self.assertEqual(result["inserted_count"], 3)
        self.assertNotIn("errors", result)


if __name__ == '__main__':
    unittest.main()
