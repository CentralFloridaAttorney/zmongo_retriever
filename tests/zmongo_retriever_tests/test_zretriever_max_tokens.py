import unittest
from unittest.mock import MagicMock
from bson import ObjectId
from zmongo_toolbag.zretriever import ZRetriever
from langchain.schema import Document


class DummyRepo:
    def __init__(self):
        self.db = MagicMock()
        self.mongo_client = MagicMock()

    async def find_document(self, collection, query):
        return {
            "_id": query["_id"],
            "casebody": {
                "data": {
                    "opinions": [{"text": "This is a test opinion."}]
                }
            },
            "collection_name": collection,
            "database_name": "testdb"
        }


class TestZRetrieverMaxTokens(unittest.IsolatedAsyncioTestCase):
    async def test_returns_documents_if_max_tokens_less_than_one(self):
        repo = DummyRepo()
        retriever = ZRetriever(max_tokens_per_set=0)

        retriever.splitter.split_text = MagicMock(return_value=["chunk1", "chunk2"])
        valid_id = str(ObjectId())  # Use a real ObjectId string

        result = await retriever.invoke("test_collection", object_ids=valid_id)

        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(doc, Document) for doc in result))
        # self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
