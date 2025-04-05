import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId

from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

class TestZMongoEmbedderEmbedText(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_repo = MagicMock()
        self.mock_repo.find_document = AsyncMock(return_value=None)
        self.mock_repo.find_one = self.mock_repo.find_document  # 👈 fix: alias with AsyncMock
        self.mock_repo.insert_document = AsyncMock()
        self.mock_repo.save_embedding = AsyncMock()

        # Create a fake OpenAI response
        class FakeEmbeddingObject:
            embedding = [0.1, 0.2, 0.3]

        class FakeEmbeddingResponse:
            data = [FakeEmbeddingObject()]

        mock_openai_client = MagicMock()
        mock_openai_client.embeddings.create = AsyncMock(return_value=FakeEmbeddingResponse())

        self.embedder = ZMongoEmbedder(repository=self.mock_repo, collection="documents")
        self.embedder.openai_client = mock_openai_client

    async def test_embed_text_success(self):
        text = "This is a test string"
        embedding = await self.embedder.embed_text(text)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.mock_repo.insert_document.assert_awaited_once()

    async def test_embed_text_cached(self):
        text = "Already cached text"
        cached_embedding = [9.9, 8.8, 7.7]
        async_mock = AsyncMock(return_value={"embedding": cached_embedding})
        self.mock_repo.find_document = async_mock
        self.mock_repo.find_one = async_mock  # 👈 same alias fix here

        embedding = await self.embedder.embed_text(text)
        self.assertEqual(embedding, cached_embedding)
        self.mock_repo.find_one.assert_awaited_once()
