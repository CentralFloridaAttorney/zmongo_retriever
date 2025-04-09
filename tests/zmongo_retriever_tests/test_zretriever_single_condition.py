import unittest
from unittest.mock import MagicMock
from bson import ObjectId
from langchain.schema import Document
from zmongo_toolbag.zretriever import ZRetriever


class TestZRetriever(unittest.TestCase):
    def setUp(self):
        # Create a mock repository
        self.mock_repo = MagicMock()
        self.mock_repo.find_document = MagicMock(return_value={
            "_id": ObjectId(),
            "database_name": "test_db",
            "collection_name": "test_collection",
            "casebody": {"data": {"opinions": [{"text": "This is a test document."}]}}
        })

        # Initialize ZRetriever with max_tokens_per_set < 1
        self.zretriever = ZRetriever(max_tokens_per_set=0)

    async def test_invoke_returns_raw_documents_when_max_tokens_per_set_less_than_1(self):
        # Call the invoke method with a test collection and document ID
        result = await self.zretriever.invoke(
            collection="test_collection",
            object_ids="test_object_id"
        )

        # Check that the result is a list of Document objects
        self.assertTrue(isinstance(result, list))
        self.assertTrue(all(isinstance(doc, Document) for doc in result))

        # Check if the content of the first document matches the expected text
        self.assertEqual(result[0].page_content, "This is a test document.")

    async def test_invoke_does_not_chunk_when_max_tokens_per_set_less_than_1(self):
        # Call the invoke method
        result = await self.zretriever.invoke(
            collection="test_collection",
            object_ids="test_object_id"
        )

        # Ensure the result is a list of documents (not chunked)
        self.assertEqual(len(result), 1)  # Only one document in this case
        self.assertEqual(result[0].page_content, "This is a test document.")


if __name__ == "__main__":
    unittest.main()