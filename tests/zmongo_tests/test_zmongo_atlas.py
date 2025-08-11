import os
import unittest
from dotenv import load_dotenv
from pathlib import Path
from bson import ObjectId

# Adjust these imports to match your project structure
from zmongo_toolbag.zmongo_atlas import ZMongoAtlas

# --- Test Configuration ---
# Load environment variables for the test database connection
load_dotenv(Path.home() / "resources" / ".env_local")
TEST_DB_NAME = "zmongo_atlas_integration_test_db"
COLLECTION_NAME = "atlas_test_coll"


class TestZMongoAtlasIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration test suite for the ZMongoAtlas class.
    This suite runs against a REAL MongoDB instance to test the full functionality,
    including the fallback mechanisms for when Atlas-specific features are not available.
    """

    @classmethod
    def setUpClass(cls):
        """Check for required environment variables before running any tests."""
        cls.MONGO_URI = os.getenv("MONGO_URI")
        if not cls.MONGO_URI:
            raise unittest.SkipTest(
                "Skipping integration tests. Please set MONGO_URI in your environment."
            )

    async def asyncSetUp(self):
        """
        Set up a new ZMongoAtlas instance and a unique collection for each test.
        """
        # Initialize ZMongoAtlas with a direct connection for testing
        self.atlas = ZMongoAtlas()
        self.atlas.db = self.atlas.db.client[TEST_DB_NAME]  # Use a dedicated test DB
        self.collection = self.atlas.db[COLLECTION_NAME]

        # Clean up the collection before each test to ensure isolation
        await self.collection.delete_many({})

    async def asyncTearDown(self):
        """
        Clean up by dropping the test collection and closing the connection.
        """
        if self.atlas:
            if self.atlas.db is not None:
                await self.atlas.db.drop_collection(COLLECTION_NAME)
            await self.atlas.close()

    async def test_vector_search_fallback_and_similarity_sorting(self):
        """
        Tests that vector_search gracefully falls back to a manual search
        and correctly sorts the results by cosine similarity.
        """
        # 1. Insert documents with known embeddings into the test database
        docs_to_insert = [
            {"_id": ObjectId(), "content": "doc_low_sim", "embedding": [[-0.9, -0.8, -0.7]]},
            {"_id": ObjectId(), "content": "doc_high_sim", "embedding": [[0.1, 0.2, 0.3]]},
        ]
        await self.collection.insert_many(docs_to_insert)

        # 2. Define a query vector that is identical to one of the documents
        query_vector = [0.1, 0.2, 0.3]

        # 3. Call the vector_search method. On a local DB, this will trigger the fallback.
        result = await self.atlas.vector_search(
            COLLECTION_NAME, query_vector, "my_index", "embedding", 5
        )

        # 4. Assertions
        self.assertTrue(result.success, f"Vector search fallback failed: {result.error}")
        self.assertEqual(len(result.data), 2, "Expected both documents to be returned by fallback.")

        # Verify that the results are sorted by score in descending order
        self.assertEqual(result.data[0]["document"]["content"], "doc_high_sim")
        self.assertAlmostEqual(result.data[0]["retrieval_score"], 1.0, places=5)

        self.assertEqual(result.data[1]["document"]["content"], "doc_low_sim")
        self.assertLess(result.data[1]["retrieval_score"], 0)  # Cosine similarity of opposite vectors is negative

    async def test_create_vector_search_index_graceful_skip(self):
        """
        Tests that creating a vector search index on a non-Atlas environment
        is gracefully skipped without raising an error.
        """
        result = await self.atlas.create_vector_search_index(
            COLLECTION_NAME, "my_index", "embedding", 1536
        )

        # Assertions
        self.assertTrue(result.success, "Method should succeed even if search is not enabled.")
        self.assertIn("Skipped index creation", result.data["message"])


if __name__ == '__main__':
    unittest.main()
