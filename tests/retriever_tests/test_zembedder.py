import unittest
import os
import asyncio
import logging
from bson import ObjectId

# --- Import the actual classes we are testing ---
from zmongo_toolbag.zembedder import ZEmbedder, CHUNK_STYLE_FIXED
from zmongo_toolbag.zmongo import ZMongo

# --- Configuration for the tests ---
MODEL_PATH_VAR = "EMBEDDING_MODEL_PATH"
MONGO_URI_VAR = "MONGO_URI"
MONGO_DB_NAME_VAR = "MONGO_DATABASE_NAME"

IS_CONFIGURED = all(os.getenv(var) for var in [MODEL_PATH_VAR, MONGO_URI_VAR, MONGO_DB_NAME_VAR])
SKIP_REASON = "Required environment variables (EMBEDDING_MODEL_PATH, MONGO_URI, MONGO_DATABASE_NAME) are not set."

# --- Test constants ---
TEST_COLLECTION = "zembedder_test_cases"
TEST_DOC_ID = ObjectId("61c0c55e0000000000000004")

BASE_TEXT = """The history of computing began long before the digital age. Early mechanical devices, like the abacus, were used for calculation for thousands of years. The true precursor to the modern computer, however, was Charles Babbage's Analytical Engine in the 19th century. Though never fully built in his lifetime, its design included an arithmetic logic unit, control flow in the form of conditional branching and loops, and integrated memory, making it the first design for a general-purpose, Turing-complete computer.

The electromechanical era followed, with devices like the Atanasoff-Berry Computer and the Harvard Mark I paving the way. The major breakthrough came with the advent of fully electronic computers during World War II. ENIAC (Electronic Numerical Integrator and Computer) was a colossal machine that used vacuum tubes instead of mechanical relays, increasing calculation speed by orders of magnitude. It was programmable, but required manual rewiring to change its operations, a tedious process that highlighted the need for a more flexible architecture."""
LONG_TEXT_FOR_CHUNKING = (BASE_TEXT + "\n\n") * 3


@unittest.skipUnless(IS_CONFIGURED, SKIP_REASON)
class TestZEmbedderIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Performs real integration tests for the ZEmbedder class.
    Requires a live MongoDB instance and a local GGUF embedding model.
    """
    embedder: ZEmbedder
    db_client: ZMongo

    @classmethod
    def setUpClass(cls):
        """Loads the model and connects to the database once for all tests."""
        print("\n--- Setting up Integration Test Suite ---")
        logging.basicConfig(level=logging.WARNING)

        print("Loading embedding model (this may take a moment)...")
        cls.db_client = ZMongo()
        cls.embedder = ZEmbedder(repository=cls.db_client, n_ctx=2048)
        print("Model loaded and database connected.")

    # THE FIX IS HERE: Renamed to asyncTearDownClass and made async
    @classmethod
    async def asyncTearDownClass(cls):
        """Cleans up the database and closes connections after all tests."""
        print("\n--- Tearing Down Integration Test Suite ---")
        if hasattr(cls, 'db_client'):
            # Now we can just await the async operation directly
            await cls.db_client.db[TEST_COLLECTION].drop()
            cls.embedder.close()
            print("Test collection dropped and connections closed.")

    async def asyncSetUp(self):
        """Runs before each test to ensure a clean slate."""
        await self.db_client.delete_document(TEST_COLLECTION, {"_id": TEST_DOC_ID})
        await self.db_client.insert_document(
            TEST_COLLECTION, {"_id": TEST_DOC_ID, "source_text": LONG_TEXT_FOR_CHUNKING}
        )

    async def test_01_document_embedding_first_time(self):
        """
        Tests that a document is correctly chunked and embedded,
        and the results are saved to the database.
        """
        print("\nRunning test_01_document_embedding_first_time...")
        result = await self.embedder.get_embedding(
            embedding_style="retrieval_document",
            collection=TEST_COLLECTION,
            document_id=TEST_DOC_ID,
            embedding_field="embeddings",
            text_field="source_text",
            chunk_style=CHUNK_STYLE_FIXED,
            chunk_size=400,
            overlap=40
        )

        self.assertTrue(result.success, "Embedding process should succeed.")
        self.assertFalse(result.data['from_cache'], "Should not be from cache on the first run.")
        self.assertEqual(result.data['dimensionality'], 768)
        self.assertTrue(result.data['vectors_count'] > 1)

        db_check = await self.db_client.find_document(TEST_COLLECTION, {"_id": TEST_DOC_ID})
        self.assertTrue(db_check.success)
        self.assertIn("embeddings", db_check.data)
        self.assertEqual(len(db_check.data['embeddings']), result.data['vectors_count'])
        print("...PASSED")

    async def test_02_document_embedding_from_cache(self):
        """
        Tests that pre-existing embeddings are correctly retrieved from the database.
        """
        print("\nRunning test_02_document_embedding_from_cache...")
        await self.embedder.get_embedding(
            embedding_style="retrieval_document",
            collection=TEST_COLLECTION, document_id=TEST_DOC_ID, embedding_field="embeddings", text_field="source_text",
            chunk_style=CHUNK_STYLE_FIXED, chunk_size=400, overlap=40
        )

        result = await self.embedder.get_embedding(
            embedding_style="retrieval_document",
            collection=TEST_COLLECTION, document_id=TEST_DOC_ID, embedding_field="embeddings"
        )

        self.assertTrue(result.success, "Retrieval from cache should succeed.")
        self.assertTrue(result.data['from_cache'], "Should be from cache on the second run.")
        self.assertEqual(result.data['dimensionality'], 768)
        self.assertTrue(result.data['vectors_count'] > 1)
        print("...PASSED")

    async def test_03_query_embedding(self):
        """
        Tests that a simple, non-persistent query embedding is generated correctly.
        """
        print("\nRunning test_03_query_embedding...")
        query = "What is a von Neumann architecture?"
        result = await self.embedder.get_embedding(text=query, as_safe_result=False)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 768)
        print("...PASSED")


if __name__ == '__main__':
    unittest.main()

