import os
import unittest
from dotenv import load_dotenv
from pathlib import Path

from zgemini import GeminiChatService
# Adjust these imports to match your project structure
from zmongo_toolbag.zmongo_service import ZMongoService

# --- Test Configuration ---
# Load environment variables for the test database and API keys
load_dotenv(Path.home() / "resources" / ".env_local")
TEST_DB_NAME = "gemini_chat_integration_test_db"
KNOWLEDGE_BASE_COLLECTION = "test_kb"
CHAT_HISTORY_COLLECTION = "chat_history"


class TestGeminiChatServiceIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration test suite for the GeminiChatService.
    This suite runs against a REAL MongoDB instance and uses the REAL Gemini API
    to test the full chat workflow.
    """

    @classmethod
    def setUpClass(cls):
        """Check for required environment variables before running any tests."""
        cls.MONGO_URI = os.getenv("MONGO_URI")
        cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not all([cls.MONGO_URI, cls.GEMINI_API_KEY]):
            raise unittest.SkipTest(
                "Skipping integration tests. Please set MONGO_URI and GEMINI_API_KEY "
                "in your environment."
            )

    async def asyncSetUp(self):
        """
        Set up a new ZMongoService and GeminiChatService instance for each test.
        """
        self.zmongo_service = ZMongoService(
            mongo_uri=self.MONGO_URI,
            db_name=TEST_DB_NAME,
            gemini_api_key=self.GEMINI_API_KEY
        )
        self.chat_service = GeminiChatService(
            zmongo_service=self.zmongo_service,
            gemini_api_key=self.GEMINI_API_KEY
        )

        # Clean up collections before each test to ensure isolation
        if self.zmongo_service.repository.db is not None:
            await self.zmongo_service.repository.db.drop_collection(KNOWLEDGE_BASE_COLLECTION)
            await self.zmongo_service.repository.db.drop_collection(CHAT_HISTORY_COLLECTION)

    async def asyncTearDown(self):
        """
        Clean up by dropping the test collections and closing the connection.
        """
        if self.zmongo_service:
            if self.zmongo_service.repository.db is not None:
                await self.zmongo_service.repository.db.drop_collection(KNOWLEDGE_BASE_COLLECTION)
                await self.zmongo_service.repository.db.drop_collection(CHAT_HISTORY_COLLECTION)
            await self.zmongo_service.close_connection()

    async def test_chat_with_context_and_history(self):
        """
        Verify the full chat workflow: adding a document, getting a relevant
        response, and saving the chat history correctly.
        """
        # 1. Add a document to the knowledge base
        doc_to_add = {
            "title": "The Planet Mars",
            "content": "Mars is the fourth planet from the Sun and is often called the 'Red Planet' because of its reddish appearance."
        }
        add_res = await self.zmongo_service.add_and_embed(
            KNOWLEDGE_BASE_COLLECTION, doc_to_add, text_field="content"
        )
        self.assertTrue(add_res.success, "Failed to add document to knowledge base.")

        # 2. Ask a question that can be answered from the context
        user_prompt = "What is the nickname for Mars?"
        # --- FIX: Unpack all three return values from the chat method ---
        llm_res, save_res, _ = await self.chat_service.chat(
            user_id="test_user_1",
            prompt=user_prompt,
            knowledge_base_collection=KNOWLEDGE_BASE_COLLECTION,
            min_score_threshold=0.1
        )

        # 3. Assertions for the LLM response
        self.assertTrue(llm_res.success, f"LLM failed to generate a response: {llm_res.error}")
        self.assertIn("Red Planet", llm_res.data, "LLM response did not include the correct information from context.")

        # 4. Assertions for the saved chat history
        self.assertTrue(save_res.success, "Failed to save chat history.")
        history_res = await self.zmongo_service.repository.find_document(CHAT_HISTORY_COLLECTION, {})
        self.assertTrue(history_res.success)
        chat_record = history_res.data

        self.assertIsNotNone(chat_record)
        self.assertEqual(chat_record["user_id"], "test_user_1")
        self.assertEqual(chat_record["prompt"], user_prompt)
        self.assertEqual(chat_record["response"], llm_res.data)
        self.assertEqual(len(chat_record["references"]), 1)
        self.assertEqual(chat_record["references"][0]["title"], "The Planet Mars")

    async def test_chat_without_sufficient_context(self):
        """
        Verify that the chat service responds appropriately when it cannot find
        relevant information in the knowledge base.
        """
        # Ask a question for which there is no context
        user_prompt = "What is the capital of France?"
        # --- FIX: Unpack all three return values from the chat method ---
        llm_res,save_res, _ = await self.chat_service.chat(
            user_id="test_user_2",
            prompt=user_prompt,
            knowledge_base_collection=KNOWLEDGE_BASE_COLLECTION
        )

        # Assert that the LLM indicates it doesn't have the information
        self.assertTrue(llm_res.success)
        self.assertIn("don't have enough information", llm_res.data.lower(),
                      "LLM should have indicated a lack of information.")

        # Verify that the chat history was still saved but has no references
        history_res = await self.zmongo_service.repository.find_document(CHAT_HISTORY_COLLECTION, {})
        self.assertTrue(history_res.success)
        chat_record = history_res.data
        self.assertEqual(chat_record["user_id"], "test_user_2")
        self.assertEqual(len(chat_record["references"]), 0)


if __name__ == '__main__':
    unittest.main()
