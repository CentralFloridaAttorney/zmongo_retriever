import unittest
from bson import ObjectId
from zmongo_toolbag import ZMongo, OpenAIModel


class TestOpenAIModelRealSave(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.model = OpenAIModel()
        self.zmongo = ZMongo()
        self.test_collection = "test_openai_results"
        self.test_id = ObjectId()

        # Pre-insert a dummy doc so we can update it
        await self.zmongo.insert_document(self.test_collection, {
            "_id": self.test_id,
            "placeholder": True
        })

    async def asyncTearDown(self):
        # Clean up
        await self.zmongo.delete_document(self.test_collection, {"_id": self.test_id})
        await self.zmongo.close()

    async def test_save_openai_result_real(self):
        result = await self.model.save_openai_result(
            collection_name=self.test_collection,
            record_id=self.test_id,
            field_name="ai_text",
            generated_text="Hello from OpenAI test!",
            extra_fields={"test_case": "real_save_test"}
        )

        self.assertTrue(result, "Expected save_openai_result to return True")

        # Verify in DB
        doc = await self.zmongo.find_document(self.test_collection, {"_id": self.test_id})
        self.assertIsNotNone(doc, f"Document with _id {self.test_id} not found in DB")
        print("Returned document:", doc)

        self.assertIn("ai_text", doc, f"Expected 'ai_text' field not found in document: {doc}")
        self.assertEqual(doc["ai_text"], "Hello from OpenAI test!")
        self.assertEqual(doc["test_case"], "real_save_test")


if __name__ == "__main__":
    unittest.main()
