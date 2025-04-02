import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId, json_util

from zconstants import EMBEDDING_MODEL
from zmongo_toolbag.zmongo import ZMongo, DEFAULT_QUERY_LIMIT
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder


import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoRepository(unittest.IsolatedAsyncioTestCase):


    async def test_delete_document(self):
        """
        Test deleting a document from the database.
        """
        collection = "test_collection"
        query = {"_id": ObjectId("65f1b6beae7cd4d4d1d3ae8d")}

        # Mock the delete_one method
        mock_result = unittest.mock.Mock(deleted_count=1)
        self.repository.db[collection].delete_one = AsyncMock(return_value=mock_result)

        # Call the method
        result = await self.repository.delete_document(collection, query)

        # Assert
        self.assertEqual(result.deleted_count, 1)
        self.repository.db[collection].delete_one.assert_awaited_once_with(filter=query)

    async def test_get_simulation_steps(self):
        """
        Test fetching simulation steps for a specific simulation.
        """
        collection = "test_collection"
        simulation_id = ObjectId("65f1b6beae7cd4d4d1d3ae8d")

        # Mock the return value of find().sort().to_list()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor  # sort() returns the cursor object for chaining
        mock_cursor.to_list = AsyncMock(
            return_value=[
                {"step": 1, "data": "Step 1"},
                {"step": 2, "data": "Step 2"}
            ]
        )

        self.repository.db[collection].find.return_value = mock_cursor

        # Call the method
        result = await self.repository.get_simulation_steps(collection, simulation_id)

        # Assert
        expected_result = [
            {"step": 1, "data": "Step 1"},
            {"step": 2, "data": "Step 2"}
        ]
        self.assertEqual(result, expected_result)

        self.repository.db[collection].find.assert_called_once_with({"simulation_id": simulation_id})
        mock_cursor.sort.assert_called_once_with("step", 1)
        mock_cursor.to_list.assert_awaited_once_with(length=None)


    def setUp(self):
        # Create the repository instance and mock the db
        self.repository = ZMongo()
        self.repository.db = MagicMock()  # Mock the database client

        # Mock the ZMongoEmbedder
        self.embedder = ZMongoEmbedder(repository=self.repository, collection="test_collection")
        self.embedder.openai_client = MagicMock()  # Mock the OpenAI client

    async def test_find_document_cache_miss(self):
        collection = "test_collection"
        query = {"_id": "123"}

        # Prepare the mocked return value
        mock_result = {"_id": ObjectId("65f1b6beae7cd4d4d1d3ae8d"), "name": "Document"}
        self.repository.db[collection].find_one = AsyncMock(return_value=mock_result)

        # Call the method under test
        result = await self.repository.find_document(collection, query)

        # Serialize both expected and actual results for consistent comparison
        expected_result = json_util.loads(json_util.dumps(mock_result))
        actual_result = json_util.loads(json_util.dumps(result))

        # Assert the result matches the serialized expected result
        self.assertEqual(actual_result, expected_result)
        self.repository.db[collection].find_one.assert_awaited_once_with(filter=query)


    async def test_insert_document(self):
        collection = "test_collection"
        document = {"name": "New Document"}
        mock_result = ObjectId("65f1b6beae7cd4d4d1d3ae8d")

        # Mock the insert_one method
        self.repository.db[collection].insert_one = AsyncMock(
            return_value=unittest.mock.Mock(inserted_id=mock_result))

        result = await self.repository.insert_document(collection, document)

        self.assertEqual(result.inserted_id, mock_result)
        self.repository.db[collection].insert_one.assert_awaited_once_with(document)


    async def test_update_document(self):
        collection = "test_collection"
        query = {"_id": ObjectId()}
        update_data = {"$set": {"name": "Updated Value"}}

        # Mock the update_one method
        mock_update_result = unittest.mock.Mock(
            matched_count=1, modified_count=1, upserted_id=None)
        self.repository.db[collection].update_one = AsyncMock(return_value=mock_update_result)

        result = await self.repository.update_document(collection, query, update_data)

        self.assertEqual(result["matchedCount"], 1)
        self.assertEqual(result["modifiedCount"], 1)
        self.assertIsNone(result["upsertedId"])
        self.repository.db[collection].update_one.assert_awaited_once_with(
            filter=query, update=update_data, upsert=False, array_filters=None)

    async def test_find_documents(self):
        collection = "test_collection"
        query = {"status": "active"}

        # Mock find().to_list()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": ObjectId(), "name": "Alice", "status": "active"},
            {"_id": ObjectId(), "name": "Bob", "status": "active"}
        ])
        self.repository.db[collection].find.return_value = mock_cursor

        result = await self.repository.find_documents(collection, query)

        self.assertEqual(len(result), 2)
        self.repository.db[collection].find.assert_called_once_with(filter=query)
        mock_cursor.to_list.assert_awaited_once_with(length=DEFAULT_QUERY_LIMIT)

    async def test_embed_text(self):
        text = "Sample text for embedding."
        expected_embedding = [0.1, 0.2, 0.3]

        # Mock the OpenAI client response
        self.embedder.openai_client.embeddings.create = AsyncMock(return_value=unittest.mock.Mock(
            data=[unittest.mock.Mock(embedding=expected_embedding)]
        ))

        result = await self.embedder.embed_text(text)
        self.assertEqual(result, expected_embedding)
        self.embedder.openai_client.embeddings.create.assert_awaited_once_with(
            model=EMBEDDING_MODEL, input=[text])

    async def test_embed_and_store(self):
        document_id = ObjectId()
        text = "Sample text for embedding."
        embedding_field = "embedding"
        expected_embedding = [0.1, 0.2, 0.3]

        # Mock methods for embedding and saving
        self.embedder.embed_text = AsyncMock(return_value=expected_embedding)
        self.repository.save_embedding = AsyncMock()

        await self.embedder.embed_and_store(document_id, text, embedding_field)

        self.embedder.embed_text.assert_awaited_once_with(text)
        self.repository.save_embedding.assert_awaited_once_with(
            self.embedder.collection, document_id, expected_embedding, embedding_field)

    async def test_clear_cache(self):
        # Populate cache
        self.repository.cache["test_collection"]["key"] = {"name": "Alice"}

        self.assertTrue(self.repository.cache["test_collection"])
        await self.repository.clear_cache()

        self.assertFalse(self.repository.cache["test_collection"])


if __name__ == "__main__":
    unittest.main()
