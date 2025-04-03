import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from langchain.schema import Document

from zmongo_retriever.zmongo_toolbag.zmongo_retriever import ZMongoRetriever


class TestZMongoRetriever(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = MagicMock()
        self.repo.find_document = AsyncMock()
        self.repo.db = MagicMock()
        self.repo.mongo_client = MagicMock()

        self.retriever = ZMongoRetriever(repository=self.repo, max_tokens_per_set=50, chunk_size=10)

    async def test_get_zdocuments_skips_missing_or_invalid(self):
        self.repo.find_document.return_value = None
        docs = await self.retriever.get_zdocuments("test", [str(ObjectId())])
        self.assertEqual(docs, [])

    async def test_get_zdocuments_valid(self):
        obj_id = ObjectId()
        self.repo.find_document.return_value = {
            "_id": obj_id,
            "casebody": {"data": {"opinions": [{"text": "Hello world. This is a test."}]}}
        }
        docs = await self.retriever.get_zdocuments("test", [str(obj_id)])
        self.assertTrue(len(docs) > 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in docs))

    async def test_invoke_returns_chunked_sets(self):
        obj_id = ObjectId()
        self.repo.find_document.return_value = {
            "_id": obj_id,
            "casebody": {"data": {"opinions": [{"text": "This is a long enough text to produce multiple chunks."}]}}
        }
        results = await self.retriever.invoke(collection="test", object_ids=[str(obj_id)])
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(chunk, Document) for chunk in results[0]))

    async def test_num_tokens_from_string(self):
        text = "hello world"
        tokens = self.retriever.num_tokens_from_string(text)
        self.assertTrue(isinstance(tokens, int))
        self.assertGreater(tokens, 0)

    async def test_get_chunk_sets_respects_token_limit(self):
        docs = [Document(page_content="a" * 10, metadata={}) for _ in range(10)]
        chunked = self.retriever.get_chunk_sets(docs)
        self.assertIsInstance(chunked, list)
        self.assertTrue(all(isinstance(group, list) for group in chunked))