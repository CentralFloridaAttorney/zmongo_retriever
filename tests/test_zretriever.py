import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from langchain.schema import Document

from zmongo_toolbag.zretriever import ZRetriever


class TestZRetriever(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = MagicMock()
        self.repo.find_document = AsyncMock()
        self.repo.db = MagicMock()
        self.repo.mongo_client = MagicMock()

        self.retriever = ZRetriever(repository=self.repo, max_tokens_per_set=20, chunk_size=10, overlap_prior_chunks=2)

    async def test_chunk_sets_with_overlap(self):
        docs = [Document(page_content="token " * 8, metadata={}) for _ in range(5)]
        chunked = self.retriever.get_chunk_sets(docs)

        self.assertIsInstance(chunked, list)
        self.assertGreater(len(chunked), 1, "Should create multiple chunk sets")

        for i in range(1, len(chunked)):
            overlap = chunked[i - 1][-self.retriever.overlap_prior_chunks:]
            next_chunk_start = chunked[i][:self.retriever.overlap_prior_chunks]
            self.assertEqual(overlap, next_chunk_start, "Incorrect chunk overlap")

    async def test_get_zdocuments_skips_missing_or_invalid(self):
        self.repo.find_document.return_value = None
        docs = await self.retriever.get_zdocuments("test", [str(ObjectId())])
        self.assertEqual(docs, [])

    async def test_get_zdocuments_valid(self):
        obj_id = ObjectId()
        self.repo.find_document.return_value = {"_id": obj_id, "text": "Hello world. This is a test."}
        docs = await self.retriever.get_zdocuments("test", [str(obj_id)], page_content_key="text")
        self.assertTrue(len(docs) > 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in docs))

    async def test_get_zdocuments_single_string_object_id(self):
        obj_id = str(ObjectId())
        self.repo.find_document.return_value = {"_id": obj_id, "text": "Testing string input."}
        docs = await self.retriever.get_zdocuments("test", obj_id, page_content_key="text")
        self.assertTrue(len(docs) > 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in docs))

    async def test_get_zdocuments_invalid_page_content_type(self):
        obj_id = ObjectId()
        self.repo.find_document.return_value = {"_id": obj_id, "text": 12345}  # Not a string
        docs = await self.retriever.get_zdocuments("test", [str(obj_id)], page_content_key="text")
        self.assertEqual(docs, [])

    async def test_invoke_returns_chunked_sets(self):
        obj_id = ObjectId()
        self.repo.find_document.return_value = {"_id": obj_id, "text": "This is a long enough text to produce multiple chunks."}
        results = await self.retriever.invoke(collection="test", object_ids=[str(obj_id)], page_content_key="text")
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(chunk, Document) for chunk in results[0]))

    async def test_num_tokens_from_string(self):
        tokens = self.retriever.num_tokens_from_string("hello world")
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)

    async def test_get_chunk_sets_respects_token_limit(self):
        docs = [Document(page_content="a" * 10, metadata={}) for _ in range(10)]
        chunked = self.retriever.get_chunk_sets(docs)
        self.assertIsInstance(chunked, list)
        self.assertTrue(all(isinstance(group, list) for group in chunked))
