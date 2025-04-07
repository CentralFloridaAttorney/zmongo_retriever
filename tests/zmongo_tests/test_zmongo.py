import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId, json_util
from zmongo_toolbag.zmongo import ZMongo, DEFAULT_QUERY_LIMIT
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
import asyncio


class TestZMongoAndEmbedder(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()
        self.repo.mongo_client = MagicMock()
        self.repo.cache.clear()

        self.embedder = ZMongoEmbedder(collection="test_collection")
        self.embedder.openai_client = MagicMock()

    async def test_find_document_cache_miss_and_hit(self):
        collection = "test"
        query = {"_id": ObjectId()}
        serialized_doc = {"_id": str(query["_id"]), "name": "test"}

        self.repo.db[collection].find_one = AsyncMock(return_value=query)
        with patch.object(ZMongo, 'serialize_document', return_value=serialized_doc):
            result = await self.repo.find_document(collection, query)
            self.assertEqual(result, serialized_doc)
            # Second call should hit cache
            cached = await self.repo.find_document(collection, query)
            self.assertEqual(cached, serialized_doc)

    async def test_find_documents(self):
        collection = "test"
        query = {"status": "ok"}
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[{"_id": 1}, {"_id": 2}])
        self.repo.db[collection].find.return_value = cursor
        result = await self.repo.find_documents(collection, query)
        self.assertEqual(len(result), 2)

    import unittest
    from unittest.mock import AsyncMock, MagicMock
    from bson import ObjectId

    class TestZMongoInsert(unittest.IsolatedAsyncioTestCase):
        async def asyncSetUp(self):
            from zmongo_toolbag.zmongo import ZMongo  # Adjust import path to your project layout
            self.repo = ZMongo()
            self.repo.db = MagicMock()

        async def test_insert_document(self):
            collection = "test"
            doc = {"name": "Alice"}
            inserted_id = ObjectId()

            mock_insert_result = MagicMock()
            mock_insert_result.inserted_id = inserted_id

            self.repo.db[collection].insert_one = AsyncMock(return_value=mock_insert_result)

            result = await self.repo.insert_document(collection, doc)

            self.assertIsInstance(result, dict)
            self.assertEqual(result["_id"], inserted_id)
            self.assertEqual(result["name"], "Alice")

    async def test_update_document_success_and_fail(self):
        collection = "test"
        query = {"x": 1}
        update = {"$set": {"x": 2}}

        mock_result = MagicMock(matched_count=1, modified_count=1, upserted_id=None)
        updated_doc = {"_id": ObjectId(), "x": 2}

        self.repo.db[collection].update_one = AsyncMock(return_value=mock_result)
        self.repo.db[collection].find_one = AsyncMock(return_value=updated_doc)

        result = await self.repo.update_document(collection, query, update)
        self.assertEqual(result["matched_count"], 1)

        # simulate failure
        self.repo.db[collection].update_one = AsyncMock(side_effect=Exception("fail"))
        result = await self.repo.update_document(collection, query, update)
        self.assertEqual(result, {})

    async def test_get_simulation_steps_valid_and_invalid(self):
        collection = "test"
        sim_id = ObjectId()
        cursor = MagicMock()
        cursor.sort.return_value = cursor
        cursor.to_list = AsyncMock(return_value=[{"step": 1}, {"step": 2}])
        self.repo.db[collection].find.return_value = cursor
        steps = await self.repo.get_simulation_steps(collection, sim_id)
        self.assertEqual(len(steps), 2)

        steps = await self.repo.get_simulation_steps(collection, "not_an_objectid")
        self.assertEqual(steps, [])

    async def test_save_embedding(self):
        collection = "test"
        doc_id = ObjectId()
        embedding = [0.1, 0.2]
        self.repo.db[collection].update_one = AsyncMock()
        await self.repo.save_embedding(collection, doc_id, embedding, "vec")
        self.repo.db[collection].update_one.assert_awaited_once()

    async def test_clear_cache(self):
        self.repo.cache["collection"]["key"] = {"x": 1}
        await self.repo.clear_cache()
        self.assertEqual(len(self.repo.cache["collection"]), 0)

    async def test_bulk_write(self):
        collection = "test"
        operations = []
        self.repo.db[collection].bulk_write = AsyncMock()
        await self.repo.bulk_write(collection, operations)

        operations = [MagicMock()]
        await self.repo.bulk_write(collection, operations)
        self.repo.db[collection].bulk_write.assert_awaited_once()

    async def test_close_connection(self):
        self.repo.mongo_client.close = MagicMock()
        await self.repo.close()
        self.repo.mongo_client.close.assert_called_once()

    async def test_embed_and_store(self):
        mock_repo = MagicMock()
        mock_repo.save_embedding = AsyncMock()

        embedder = ZMongoEmbedder(
            collection="test_collection",
            repository=mock_repo  # ✅ inject the mocked repository
        )
        embedder.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])  # mock embedding

        doc_id = ObjectId()
        text = "test string for embedding"

        await embedder.embed_and_store(document_id=doc_id, text=text)

        mock_repo.save_embedding.assert_awaited_once_with("test_collection", doc_id, [0.1, 0.2, 0.3], "embedding")

    async def test_delete_all_documents(self):
        collection = "test"
        deleted_count = 42

        mock_result = MagicMock(deleted_count=deleted_count)
        self.repo.db[collection].delete_many = AsyncMock(return_value=mock_result)

        result = await self.repo.delete_all_documents(collection)
        self.repo.db[collection].delete_many.assert_awaited_once_with({})
        self.assertEqual(result, deleted_count)

    async def test_insert_documents_logs_error_on_failure(self):
        collection = "test"
        docs = [{"name": "Doc 1"}, {"name": "Doc 2"}]

        # Simulate insert_many throwing an exception
        self.repo.db[collection].insert_many = AsyncMock(side_effect=Exception("Batch insert failed"))

        with self.assertLogs("zmongo_toolbag.zmongo", level="ERROR") as cm:
            result = await self.repo.insert_documents(collection, docs, batch_size=2)

        self.assertEqual(result["inserted_count"], 0)
        self.assertIn("Batch insert failed", "".join(cm.output))

    async def test_insert_documents_cache_and_ids(self):
        collection = "test"
        docs = [{"name": "Alice"}, {"name": "Bob"}]
        inserted_ids = [ObjectId(), ObjectId()]

        # Simulate insert_many returning inserted_ids
        mock_result = MagicMock()
        mock_result.inserted_ids = inserted_ids

        self.repo.db[collection].insert_many = AsyncMock(return_value=mock_result)

        with patch.object(self.repo, "serialize_document", side_effect=lambda d: d):
            result = await self.repo.insert_documents(collection, docs)

        # Verify inserted_count matches
        self.assertEqual(result["inserted_count"], 2)

        normalized = self.repo._normalize_collection_name(collection)
        for doc, _id in zip(docs, inserted_ids):
            self.assertEqual(doc["_id"], _id)
            key = self.repo._generate_cache_key({"_id": str(_id)})
            self.assertIn(key, self.repo.cache[normalized])
            self.assertEqual(self.repo.cache[normalized][key]["_id"], _id)

    @patch("zmongo_toolbag.zmongo.logger")
    async def test_insert_document_handles_exception(self, mock_logger):
        collection_name = "test_collection"
        document = {"invalid": True}

        # Patch the internal DB collection to raise an error on insert
        mocked_collection = MagicMock()
        mocked_collection.insert_one = AsyncMock(side_effect=Exception("Mocked insertion failure"))

        # Patch self.repo.db[collection_name] to return the mocked collection
        self.repo.db.__getitem__ = MagicMock(return_value=mocked_collection)

        # Act
        result = await self.repo.insert_document(collection_name, document)

        # Assert
        self.assertIsNone(result)
        mock_logger.error.assert_called_once()
        self.assertIn("Error inserting document into", mock_logger.error.call_args[0][0])

    async def test_zmongo_initialization_no_env(self):
        """
        Test that ZMongo logs warnings and uses defaults when .env is missing or empty.
        """
        # Example of forcing environment vars to be missing
        with patch.dict(os.environ, {}, clear=True):
            with patch("zmongo_toolbag.zmongo.logger.warning") as mock_warning:
                zmongo_instance = ZMongo()

        self.assertEqual(zmongo_instance.MONGO_URI, "mongodb://127.0.0.1:27017")
        self.assertEqual(zmongo_instance.MONGO_DB_NAME, "documents")

        # Confirm the two warnings were logged
        mock_warning.assert_any_call(
            "⚠️ MONGO_URI is not set in .env. Defaulting to 'mongodb://127.0.0.1:27017'"
        )
        mock_warning.assert_any_call(
            "❌ MONGO_DATABASE_NAME is not set in .env. Defaulting to 'documents'"
        )

if __name__ == "__main__":
    unittest.main()
