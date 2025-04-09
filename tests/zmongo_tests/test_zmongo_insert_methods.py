import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from zmongo_toolbag.zmongo import ZMongo
from datetime import datetime


class TestZMongoInsertMethods(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.zmongo = ZMongo()
        self.zmongo.db = MagicMock()
        self.zmongo.sync_db = MagicMock()

    async def test_insert_documents_async_success(self):
        collection = "test_collection"
        docs = [{"a": 1}, {"b": 2}]
        inserted_ids = [1, 2]

        # Mock successful batch insert
        mock_result = MagicMock(inserted_ids=inserted_ids)
        self.zmongo.db[collection].insert_many = AsyncMock(return_value=mock_result)

        result = await self.zmongo.insert_documents(collection, docs, batch_size=2, use_cache=False)
        self.assertEqual(result["inserted_count"], 2)
        self.assertNotIn("errors", result)

    async def test_insert_documents_async_failure(self):
        collection = "test_collection"
        docs = [{"a": 1}]
        self.zmongo.db[collection].insert_many = AsyncMock(side_effect=Exception("batch insert error"))

        result = await self.zmongo.insert_documents(collection, docs)
        self.assertEqual(result["inserted_count"], 0)
        self.assertIn("errors", result)
        self.assertTrue(any("batch insert error" in err for err in result["errors"]))

    async def test_insert_documents_empty(self):
        result = await self.zmongo.insert_documents("any_collection", [])
        self.assertEqual(result, {"inserted_count": 0})

    def test_insert_documents_sync_success(self):
        collection = "sync_test"
        docs = [{"x": 1}, {"y": 2}]
        mock_result = MagicMock(inserted_ids=[1, 2])
        self.zmongo.sync_db[collection].insert_many.return_value = mock_result

        result = self.zmongo.insert_documents_sync(collection, docs)
        self.assertEqual(result["inserted_count"], 2)
        self.assertNotIn("errors", result)

    def test_insert_documents_sync_failure(self):
        collection = "sync_fail"
        docs = [{"a": 1}]
        self.zmongo.sync_db[collection].insert_many.side_effect = Exception("sync fail")

        result = self.zmongo.insert_documents_sync(collection, docs)
        self.assertEqual(result["inserted_count"], 0)
        self.assertIn("errors", result)
        self.assertTrue(any("sync fail" in err for err in result["errors"]))

    def test_insert_documents_sync_empty(self):
        result = self.zmongo.insert_documents_sync("irrelevant", [])
        self.assertEqual(result, {"inserted_count": 0})

    def test_log_training_metrics_success(self):
        self.zmongo.sync_db["training_metrics"].insert_one = MagicMock()
        metrics = {"loss": 0.01, "accuracy": 0.99}
        self.zmongo.log_training_metrics(metrics)
        self.zmongo.sync_db["training_metrics"].insert_one.assert_called_once()
        inserted_doc = self.zmongo.sync_db["training_metrics"].insert_one.call_args[0][0]
        self.assertIn("timestamp", inserted_doc)
        self.assertIn("loss", inserted_doc)

    def test_log_training_metrics_failure(self):
        self.zmongo.sync_db["training_metrics"].insert_one.side_effect = Exception("write error")
        with patch("zmongo_toolbag.zmongo.logger") as mock_logger:
            self.zmongo.log_training_metrics({"foo": "bar"})
            mock_logger.error.assert_called_with("Failed to log training metrics: write error")


if __name__ == "__main__":
    unittest.main()
