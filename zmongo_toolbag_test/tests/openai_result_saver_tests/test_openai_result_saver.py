import unittest
from bson.objectid import ObjectId

from models import OpenAIResultSaver
from zmongo_toolbag.zmongo import ZMongo


class TestOpenAIResultSaverReal(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.collection = "test_openai_result_saver"
        self.test_id = ObjectId()
        self.test_id_str = str(self.test_id)

    async def asyncSetUp(self):
        self.zmongo = ZMongo()
        await self.zmongo.insert_document(self.collection, {
            "_id": self.test_id,
            "initial": True
        })

    async def asyncTearDown(self):
        fresh = ZMongo()
        await fresh.delete_document(self.collection, {"_id": self.test_id})
        await fresh.close()

    async def test_openai_result_saver_save_success(self):
        saver = OpenAIResultSaver()

        result = await saver.save(
            collection_name=self.collection,
            record_id=self.test_id_str,  # str form
            field_name="ai_text",
            generated_text="Hello, world!",
            extra_fields={"source": "unit_test"}
        )

        self.assertTrue(result, "Expected save() to return True")

        # Confirm DB update using a fresh ZMongo
        fresh = ZMongo()
        doc = await fresh.find_document(self.collection, {"_id": self.test_id})
        self.assertEqual(doc["ai_text"], "Hello, world!")
        self.assertEqual(doc["source"], "unit_test")
        await fresh.close()


if __name__ == "__main__":
    unittest.main()
