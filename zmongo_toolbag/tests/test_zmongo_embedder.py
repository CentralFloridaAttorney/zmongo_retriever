import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from bson.objectid import ObjectId
import logging
import io
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder


class TestZMongoEmbedder(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for ZMongoEmbedder class.
    """

    def setUp(self):
        """
        Set up the test environment by mocking dependencies.
        """
        # Mock the repository
        self.repository = MagicMock()

        # Mock the OpenAI client
        self.mock_openai_client = MagicMock()

        # Patch environment variables
        self.openai_api_key = "mock_api_key"
        self.embedding_model = "mock_embedding_model"
        patcher = patch.dict('os.environ', {
            "OPENAI_API_KEY": self.openai_api_key,
            "EMBEDDING_MODEL": self.embedding_model
        })
        self.addCleanup(patcher.stop)
        patcher.start()

        # Create an instance of ZMongoEmbedder with mocks
        self.embedder = ZMongoEmbedder(repository=self.repository, collection="test_collection")
        self.embedder.openai_client = self.mock_openai_client  # Inject the mocked OpenAI client

        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)  # Re-enable logging after tests

    async def test_embed_text_success(self):
        """
        Test embed_text with valid text and ensure it returns the expected embedding.
        """
        # Mock OpenAI API response
        mock_embedding = [1.0, 2.0, 3.0, 4.0]
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=mock_embedding)])
        )

        # Call the method
        result = await self.embedder.embed_text("Test text")

        # Assertions
        self.mock_openai_client.embeddings.create.assert_awaited_once_with(
            model=self.embedding_model, input=["Test text"]
        )
        self.assertEqual(result, mock_embedding)

    async def test_embed_text_invalid_input(self):
        """
        Test embed_text with invalid input (e.g., empty string) and ensure it raises a ValueError.
        """
        with self.assertRaises(ValueError) as context:
            await self.embedder.embed_text("")

        self.assertEqual(str(context.exception), "text must be a non-empty string")

    async def test_embed_text_api_error(self):
        """
        Test embed_text when the OpenAI API raises an error and ensure it is handled correctly.
        """
        self.mock_openai_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

        with self.assertRaises(Exception) as context:
            await self.embedder.embed_text("Some text")

        self.mock_openai_client.embeddings.create.assert_awaited_once()
        self.assertEqual(str(context.exception), "API Error")

    async def test_embed_text_invalid_openai_response(self):
        """
        Test embed_text when OpenAI API returns an unexpected or invalid response format.
        """
        # Mock OpenAI API response with invalid structure
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[])  # No embedding data
        )

        with self.assertRaises(ValueError) as context:
            await self.embedder.embed_text("Test text")

        self.assertEqual(str(context.exception), "Invalid response format from OpenAI API: missing embedding data")

    async def test_embed_and_store_success(self):
        """
        Test embed_and_store with valid inputs and ensure embedding is stored in the database.
        """
        # Mock OpenAI embedding result
        mock_embedding = [1.0, 2.0, 3.0, 4.0]
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=mock_embedding)])
        )

        # Mock save_embedding method in the repository
        self.repository.save_embedding = AsyncMock()

        # Document ID and text
        document_id = ObjectId()
        text = "Sample text"

        # Call the method
        await self.embedder.embed_and_store(document_id, text)

        # Assertions
        self.mock_openai_client.embeddings.create.assert_awaited_once_with(
            model=self.embedding_model, input=[text]
        )
        self.repository.save_embedding.assert_awaited_once_with(
            "test_collection", document_id, mock_embedding, "embedding"
        )

    async def test_embed_and_store_invalid_document_id(self):
        """
        Test embed_and_store with an invalid document_id (not ObjectId).
        """
        with self.assertRaises(ValueError) as context:
            await self.embedder.embed_and_store("invalid_id", "Test text")

        self.assertEqual(str(context.exception), "document_id must be an instance of ObjectId")

    async def test_embed_and_store_repository_error(self):
        """
        Test embed_and_store when repository.save_embedding throws an exception.
        """
        # Mock OpenAI embedding result
        mock_embedding = [1.0, 2.0, 3.0, 4.0]
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=mock_embedding)])
        )

        # Mock repository.save_embedding to raise an exception
        self.repository.save_embedding = AsyncMock(side_effect=Exception("Database Error"))

        # Document ID and text
        document_id = ObjectId()
        text = "Sample text"

        # Call the method and check exception
        with self.assertRaises(Exception) as context:
            await self.embedder.embed_and_store(document_id, text)

        self.mock_openai_client.embeddings.create.assert_awaited_once()  # Ensure API was called
        self.repository.save_embedding.assert_awaited_once()  # Ensure repository was called
        self.assertEqual(str(context.exception), "Database Error")
