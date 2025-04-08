import unittest
from bson import ObjectId
from langchain.schema import Document

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zretriever import ZRetriever


class TestZRetrieverWithRealDB(unittest.IsolatedAsyncioTestCase):
    """
    Test class for ZRetriever that uses real ZMongo interactions (no mocks).

    IMPORTANT:
      - Requires MongoDB access via MONGO_URI and MONGO_DATABASE_NAME environment variables,
        or defaults to "mongodb://127.0.0.1:27017" and db name "documents".
      - Will insert/delete documents in "test_collection" for testing, so use caution if
        you have existing data under that name.
    """

    async def asyncSetUp(self):
        """
        Runs once before each test, creating a real ZMongo instance
        and clearing the 'test_collection' to ensure a clean slate.
        """
        self.zmongo = ZMongo()  # real connection
        self.retriever = ZRetriever(
            overlap_prior_chunks=2,
            max_tokens_per_set=4096,
            chunk_size=512
        )

        # Clean up (or create) the test_collection before each test
        await self.zmongo.delete_all_documents("test_collection")

    async def asyncTearDown(self):
        """
        Runs after each test, cleaning up any inserted documents.
        Closes the MongoDB connections.
        """
        await self.zmongo.delete_all_documents("test_collection")
        await self.zmongo.close()

    async def test_get_zdocuments_valid(self):
        """
        Insert a valid doc with string text and verify get_zdocuments
        retrieves it as a Document object.
        """
        obj_id = ObjectId()
        await self.zmongo.insert_document("test_collection", {
            "_id": obj_id,
            "text": "Hello world. This is a test."
        })

        docs = await self.retriever.get_zdocuments(
            collection="test_collection",
            object_ids=[str(obj_id)],
            page_content_key="text"
        )

        self.assertTrue(len(docs) > 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in docs))
        self.assertEqual(docs[0].page_content, "Hello world. This is a test.")

    async def test_invoke_returns_raw_documents_when_max_tokens_per_set_less_than_1(self):
        """
        If max_tokens_per_set < 1, invoke() should return the raw documents (no chunking).
        """
        # Change retriever config on the fly:
        self.retriever.max_tokens_per_set = 0

        obj_id = ObjectId()
        await self.zmongo.insert_document("test_collection", {
            "_id": obj_id,
            "text": "This is a test document."
        })

        result = await self.retriever.invoke(
            collection="test_collection",
            object_ids=str(obj_id),
            page_content_key="text"
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "This is a test document.")

    async def test_get_zdocuments_invalid_page_content_type(self):
        """
        Insert a doc with non-string 'text' field;
        ZRetriever should skip it, returning an empty list.
        """
        obj_id = ObjectId()
        await self.zmongo.insert_document("test_collection", {
            "_id": obj_id,
            "text": 12345  # Not a string
        })

        docs = await self.retriever.get_zdocuments(
            collection="test_collection",
            object_ids=str(obj_id),
            page_content_key="text"
        )

        self.assertEqual(docs, [])


if __name__ == "__main__":
    unittest.main()
