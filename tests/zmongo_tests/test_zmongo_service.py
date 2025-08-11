import os
import unittest
from dotenv import load_dotenv
from pathlib import Path

# Assuming zmongo_service.py is in the same directory or accessible in the path
from zmongo_service import ZMongoService

# ---
# IMPORTANT: Before running, ensure you have a .env_local file configured
# with a MONGO_URI, a dedicated MONGO_DATABASE_NAME for testing,
# and a valid GEMINI_API_KEY.
# ---
load_dotenv(Path.home() / "resources" / ".env_local")


class TestZMongoServiceIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration test suite for the ZMongoService class.
    This suite runs against a REAL MongoDB instance and uses the REAL Gemini API
    to test the full add, embed, and search workflow.
    """

    @classmethod
    def setUpClass(cls):
        """Check for required environment variables before running any tests."""
        cls.MONGO_URI = os.getenv("MONGO_URI")
        cls.DB_NAME = os.getenv("MONGO_DATABASE_NAME")
        cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not all([cls.MONGO_URI, cls.DB_NAME, cls.GEMINI_API_KEY]):
            raise unittest.SkipTest(
                "Skipping integration tests. Please set MONGO_URI, MONGO_DATABASE_NAME, "
                "and GEMINI_API_KEY in your .env_local file."
            )

    async def asyncSetUp(self):
        """
        Set up a new ZMongoService instance and a unique collection for each test.
        """
        self.service = ZMongoService(
            mongo_uri=self.MONGO_URI,
            db_name=self.DB_NAME,
            gemini_api_key=self.GEMINI_API_KEY
        )
        self.collection_name = self._testMethodName
        # Clean up the collection before each test to ensure isolation
        if self.service.repository.db is not None:
            await self.service.repository.db.drop_collection(self.collection_name)

    async def asyncTearDown(self):
        """
        Clean up by dropping the test collection and closing the connection.
        """
        if self.service:
            if self.service.repository.db is not None:
                await self.service.repository.db.drop_collection(self.collection_name)
            await self.service.close_connection()

    async def test_add_and_embed_new_document(self):
        """
        Verify that a new document is successfully inserted and embedded.
        """
        doc_to_add = {
            "title": "The Art of Programming",
            "content": "Programming is the process of creating a set of instructions that tell a computer how to perform a task."
        }
        text_field = "content"
        embedding_field = "embeddings"

        # Add the document
        result = await self.service.add_and_embed(
            self.collection_name, doc_to_add, text_field, embedding_field
        )

        # Assertions for the service response
        self.assertTrue(result.success, f"Service call failed: {result.error}")
        self.assertFalse(result.data.get("existed"), "Document should not have existed previously.")
        self.assertIn("inserted_id", result.data)
        doc_id = result.data["inserted_id"]

        # Verify directly in the database
        retrieved_doc_res = await self.service.repository.find_document(
            self.collection_name, {"_id": doc_id}
        )
        self.assertTrue(retrieved_doc_res.success)
        retrieved_doc = retrieved_doc_res.data
        self.assertIsNotNone(retrieved_doc)
        self.assertIn(embedding_field, retrieved_doc)
        self.assertIsInstance(retrieved_doc[embedding_field], list)
        self.assertGreater(len(retrieved_doc[embedding_field]), 0)
        self.assertIsInstance(retrieved_doc[embedding_field][0], list)
        self.assertIsInstance(retrieved_doc[embedding_field][0][0], float)

    async def test_add_and_embed_duplicate_document(self):
        """
        Verify that adding the same document twice does not create a duplicate entry.
        """
        doc_to_add = {
            "title": "Data Science",
            "content": "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics."
        }
        text_field = "content"

        # Add the document for the first time
        res1 = await self.service.add_and_embed(self.collection_name, doc_to_add, text_field)
        self.assertTrue(res1.success, f"First add_and_embed failed: {res1.error}")
        self.assertFalse(res1.data.get("existed"))
        first_id = res1.data.get("inserted_id")

        # Add the exact same document again
        res2 = await self.service.add_and_embed(self.collection_name, doc_to_add, text_field)
        self.assertTrue(res2.success, f"Second add_and_embed failed: {res2.error}")
        self.assertTrue(res2.data.get("existed"), "Service should report that the document already existed.")
        second_id = res2.data.get("inserted_id")

        # The IDs should be the same
        self.assertEqual(first_id, second_id)

        # Verify that only one document exists in the collection
        count_res = await self.service.repository.count_documents(self.collection_name, {})
        self.assertEqual(count_res.data["count"], 1)

    async def test_end_to_end_search(self):
        """
        Test the full workflow: add a document, search for it with a relevant
        query, and verify the results.
        """
        doc_to_add = {
            "title": "About The Sunshine State",
            "content": "Florida is a state in the Southeastern region of the United States, known for its warm climate and beautiful beaches."
        }
        content_field = "content"

        # Add the document
        add_res = await self.service.add_and_embed(self.collection_name, doc_to_add, content_field)
        self.assertTrue(add_res.success, f"Adding document for search test failed: {add_res.error}")

        # Perform a search with a semantically similar query
        query = "Where can I find a place with nice weather and coasts?"
        search_results = await self.service.search(
            self.collection_name, query, content_field=content_field, similarity_threshold=0.7
        )

        # Assertions for the search results
        self.assertIsInstance(search_results, list)
        self.assertGreater(len(search_results), 0, "Expected at least one relevant document.")

        top_result = search_results[0]
        self.assertEqual(top_result.page_content, doc_to_add["content"])
        self.assertEqual(top_result.metadata.get("title"), doc_to_add["title"])
        self.assertGreater(top_result.metadata.get("retrieval_score"), 0.7)


if __name__ == '__main__':
    unittest.main()
