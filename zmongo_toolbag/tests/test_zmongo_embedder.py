import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from bson.objectid import ObjectId
import logging

from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder


class TestZMongoEmbedder(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for ZMongoEmbedder class.
    """

    def setUp(self):
        """
        Set up the test environment by mocking dependencies.
        """
        self.repository = AsyncMock()  # Corrected for async compatibility
        self.mock_openai_client = MagicMock()

        patcher = patch.dict('os.environ', {
            "OPENAI_API_KEY": "mock_api_key",
            "EMBEDDING_MODEL": "mock_embedding_model"
        })
        self.addCleanup(patcher.stop)
        patcher.start()

        self.embedder = ZMongoEmbedder(repository=self.repository, collection="test_collection")
        self.embedder.openai_client = self.mock_openai_client

        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    async def test_embed_text_success(self):
        mock_embedding = [1.0, 2.0, 3.0, 4.0]
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=mock_embedding)])
        )

        result = await self.embedder.embed_text("Test text")

        self.mock_openai_client.embeddings.create.assert_awaited_once_with(
            model="mock_embedding_model", input=["Test text"]
        )
        self.assertEqual(result, mock_embedding)

    async def test_embed_text_invalid_input(self):
        with self.assertRaises(ValueError):
            await self.embedder.embed_text("")

    async def test_embed_text_api_error(self):
        self.mock_openai_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))
        with self.assertRaises(Exception):
            await self.embedder.embed_text("Some text")

    async def test_embed_text_invalid_openai_response(self):
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[])
        )
        with self.assertRaises(ValueError):
            await self.embedder.embed_text("Test text")

    async def test_embed_and_store_success(self):
        mock_embedding = [1.0, 2.0, 3.0, 4.0]
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=mock_embedding)])
        )
        self.repository.save_embedding = AsyncMock()

        document_id = ObjectId()
        text = "Sample text"

        await self.embedder.embed_and_store(document_id, text)

        self.mock_openai_client.embeddings.create.assert_awaited_once_with(
            model="mock_embedding_model", input=[text]
        )
        self.repository.save_embedding.assert_awaited_once_with(
            "test_collection", document_id, mock_embedding, "embedding"
        )

    async def test_embed_and_store_invalid_document_id(self):
        with self.assertRaises(ValueError):
            await self.embedder.embed_and_store("invalid_id", "Test text")

    async def test_embed_and_store_repository_error(self):
        mock_embedding = [1.0, 2.0, 3.0, 4.0]
        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=mock_embedding)])
        )
        self.repository.save_embedding = AsyncMock(side_effect=Exception("Database Error"))

        with self.assertRaises(Exception):
            await self.embedder.embed_and_store(ObjectId(), "Sample text")

    async def test_embed_text_rejects_non_string_input(self):
        with self.assertRaises(ValueError):
            await self.embedder.embed_text(1234)

    async def test_embed_text_missing_embedding_field(self):
        class NoEmbedding:
            pass

        self.mock_openai_client.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[NoEmbedding()])
        )

        with self.assertRaises(ValueError):
            await self.embedder.embed_text("Some valid text")

    async def test_embed_text_empty_or_non_string_input(self):
        invalid_inputs = [None, "", 123, [], {}, 0.0]
        for invalid in invalid_inputs:
            with self.subTest(input=invalid):
                with self.assertRaises(ValueError):
                    await self.embedder.embed_text(invalid)

    async def test_embed_and_store_invalid_text(self):
        invalid_inputs = [None, "", 123, [], {}]
        valid_id = ObjectId()
        for bad_text in invalid_inputs:
            with self.subTest(text=bad_text):
                with self.assertRaises(ValueError):
                    await self.embedder.embed_and_store(valid_id, bad_text)

    async def test_embed_and_store_invalid_text_guard_clause(self):
        document_id = ObjectId()
        self.embedder.embed_text = AsyncMock()

        invalid_inputs = [None, "", 123, [], {}]
        for bad_text in invalid_inputs:
            with self.subTest(text=bad_text):
                with self.assertRaises(ValueError):
                    await self.embedder.embed_and_store(document_id, bad_text)
                self.embedder.embed_text.assert_not_called()
