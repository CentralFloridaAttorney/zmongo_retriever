import unittest
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

class TestZMongoAndEmbedder(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repository = MagicMock()
        self.repository.find_document = AsyncMock(return_value=None)
        self.repository.insert_document = AsyncMock()
        self.repository.save_embedding = AsyncMock()

        self.embedder = ZMongoEmbedder(collection="test_collection", repository=self.repository)
        self.embedder.openai_client = MagicMock()
        self.embedder.openai_client.embeddings.create = AsyncMock()

    async def test_embed_text(self):
        text = "sample"
        embedding = [0.1, 0.2, 0.3]
        self.embedder.openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=embedding)]
        )

        result = await self.embedder.embed_text(text)
        self.assertEqual(result, embedding)
