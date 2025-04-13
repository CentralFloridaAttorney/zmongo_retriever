import unittest
from unittest.mock import AsyncMock, MagicMock, patch
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

        self.embedder = ZMongoEmbedder(repository=self.repo, collection="test_collection")
        self.embedder.openai_client = MagicMock()

    async def test_find_documents(self):
        collection = "test"
        query = {"status": "ok"}
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[{"_id": 1}, {"_id": 2}])
        self.repo.db[collection].find.return_value = cursor
        result = await self.repo.find_documents(collection, query)
        self.assertEqual(len(result), 2)

    async def test_insert_document(self):
        collection = "test"
        doc = {"name": "Alice"}
        inserted_id = ObjectId()
        self.repo.db[collection].insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))
        result = await self.repo.insert_document(collection, doc)
        self.assertEqual(result.inserted_id, inserted_id)

    async def test_update_document_success_and_fail(self):
        collection = "test"
        query = {"x": 1}
        update = {"$set": {"x": 2}}

        mock_result = MagicMock(matched_count=1, modified_count=1, upserted_id=None)
        updated_doc = {"_id": ObjectId(), "x": 2}

        # ✅ Mock collection
        mock_collection = MagicMock()
        mock_collection.update_one = AsyncMock(return_value=mock_result)
        mock_collection.find_one = AsyncMock(return_value=updated_doc)

        # ✅ Inject collection into db
        self.repo.db = MagicMock()
        self.repo.db.__getitem__.return_value = mock_collection

        # ✅ SUCCESS CASE
        result = await self.repo.update_document(collection, query, update)
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(result.modified_count, 1)
        self.assertIsNone(result.upserted_id)

        # ✅ FAILURE CASE
        failing_collection = MagicMock()
        failing_collection.update_one = AsyncMock(side_effect=Exception("fail"))
        self.repo.db.__getitem__.return_value = failing_collection

        with self.assertRaises(Exception) as cm:
            await self.repo.update_document(collection, query, update)
        self.assertIn("fail", str(cm.exception))

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
        document_id = ObjectId()
        text = "embed me"
        embedding_field = "embedding"
        embedding = [0.1, 0.2, 0.3]
        self.embedder.embed_text = AsyncMock(return_value=embedding)
        self.repo.save_embedding = AsyncMock()
        await self.embedder.embed_and_store(document_id, text, embedding_field)
        self.repo.save_embedding.assert_awaited_once()

    async def test_delete_all_documents(self):
        collection = "test"
        deleted_count = 42

        mock_result = MagicMock(deleted_count=deleted_count)
        self.repo.db[collection].delete_many = AsyncMock(return_value=mock_result)

        result = await self.repo.delete_all_documents(collection)
        self.repo.db[collection].delete_many.assert_awaited_once_with({})
        self.assertEqual(result, deleted_count)

if __name__ == "__main__":
    unittest.main()
