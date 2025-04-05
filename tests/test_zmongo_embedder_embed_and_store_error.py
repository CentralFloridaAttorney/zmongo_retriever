import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

class TestZMongoEmbedderEmbedAndStoreError(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = MagicMock()
        self.repo.save_embedding = AsyncMock()
        self.embedder = ZMongoEmbedder(repository=self.repo, collection="documents")

    async def test_embed_and_store_raises_exception_and_logs(self):
        # Simulate a valid ObjectId and text
        fake_id = ObjectId()
        fake_text = "Text that will fail"

        # Patch embed_text to raise an exception
        self.embedder.embed_text = AsyncMock(side_effect=RuntimeError("simulated OpenAI failure"))

        with self.assertLogs("zmongo_toolbag.zmongo_embedder", level="ERROR") as log_ctx:
            with self.assertRaises(RuntimeError) as ctx:
                await self.embedder.embed_and_store(fake_id, fake_text)

        self.assertIn("simulated OpenAI failure", str(ctx.exception))
        self.assertTrue(any("Failed to embed and store text" in msg for msg in log_ctx.output))
