import unittest
from unittest.mock import MagicMock
from bson import ObjectId
from langchain.schema import Document
from zmongo_toolbag.zretriever import ZRetriever


class TestZRetriever(unittest.TestCase):
    def setUp(self):
        """
        Create a mock repository that simulates find_document returning
        a single valid document with an embedded string.
        """
        self.mock_repo = MagicMock()
        self.mock_repo.find_document = MagicMock(return_value={
            "_id": ObjectId(),
            "database_name": "test_db",
            "collection_name": "test_collection",
            "casebody": {"data": {"opinions": [{"text": "This is a test document."}]}}
        })

        # Initialize ZRetriever with max_tokens_per_set < 1 so it should return raw documents
        self.zretriever = ZRetriever(max_tokens_per_set=0)

    async def test_invoke_returns_raw_documents_when_max_tokens_per_set_less_than_1(self):
        """
        Ensures that when max_tokens_per_set < 1, invoke() returns the raw (unchunked) documents.
        """
        # Call invoke with a test collection and document ID
        result = await self.zretriever.invoke(
            collection="test_collection",
            object_ids="test_object_id"
        )

        # Verify the result is a list of Document objects
        self.assertIsInstance(result, list, "Expected a list of documents")
        self.assertTrue(all(isinstance(doc, Document) for doc in result), "All items should be Document objects")
        self.assertEqual(result[0].page_content, "This is a test document.", "Content should match the mock return")

    async def test_invoke_does_not_chunk_when_max_tokens_per_set_less_than_1(self):
        """
        Confirms that no chunking occurs when max_tokens_per_set < 1,
        ensuring only a single Document is returned.
        """
        # Call invoke again under the same conditions
        result = await self.zretriever.invoke(
            collection="test_collection",
            object_ids="test_object_id"
        )

        # We expect one Document in the list (no chunking happened)
        self.assertIsInstance(result, list, "Expected a list of documents")
        self.assertEqual(len(result), 1, "Only one document should be returned")
        self.assertEqual(result[0].page_content, "This is a test document.", "Content should match the mock return")


if __name__ == "__main__":
    unittest.main()
