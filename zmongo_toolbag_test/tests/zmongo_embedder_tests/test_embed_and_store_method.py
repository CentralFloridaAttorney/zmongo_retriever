import unittest
from unittest.mock import AsyncMock, patch
from bson.objectid import ObjectId

from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zmongo import ZMongo


class TestEmbedAndStoreMethod(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.embedder = ZMongoEmbedder(repository=self.repo, collection="documents")

    async def asyncTearDown(self):
        await self.repo.close()

    async def test_embed_and_store_successful(self):
        test_id = ObjectId()
        test_text = "Embedding this for legal AI systems"
        fake_embedding = [0.1, 0.2, 0.3]

        self.embedder.embed_text = AsyncMock(return_value=fake_embedding)
        self.repo.save_embedding = AsyncMock(return_value=None)

        await self.embedder.embed_and_store(test_id, test_text)

        self.embedder.embed_text.assert_awaited_once_with(test_text)
        self.repo.save_embedding.assert_awaited_once_with(
            "documents", test_id, fake_embedding, "embedding"
        )

    async def test_embed_and_store_invalid_objectid(self):
        with self.assertRaises(ValueError) as ctx:
            await self.embedder.embed_and_store("not-an-oid", "text")
        self.assertIn("document_id must be an instance of ObjectId", str(ctx.exception))

    async def test_embed_and_store_invalid_text(self):
        with self.assertRaises(ValueError) as ctx:
            await self.embedder.embed_and_store(ObjectId(), "")
        self.assertIn("text must be a non-empty string", str(ctx.exception))
