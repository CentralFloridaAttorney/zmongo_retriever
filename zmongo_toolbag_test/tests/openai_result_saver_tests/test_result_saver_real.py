# zmongo_toolbag/tests/base_result_saver_test/test_result_saver_real.py

import unittest
from bson.objectid import ObjectId

from zmongo_toolbag import ZMongo, OpenAIResultSaver


class TestOpenAIResultSaverReal(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.zmongo = ZMongo()  # ✅ Provide external instance
        self.saver = OpenAIResultSaver(zmongo=self.zmongo)
        self.collection = "test_openai_result_saver"
        self.test_id = ObjectId()

        await self.zmongo.insert_document(self.collection, {
            "_id": self.test_id,
            "original": True
        })

    async def asyncTearDown(self):
        await self.zmongo.delete_document(self.collection, {"_id": self.test_id})
        await self.zmongo.close()  # ✅ Clean up manually

    async def test_save_updates_document(self):
        result = await self.saver.save(
            collection_name=self.collection,
            record_id=str(self.test_id),
            field_name="generated_text",
            generated_text="Hello from real test!",
            extra_fields={"unit": "integration"}
        )
        self.assertTrue(result)

        doc = await self.zmongo.find_document(self.collection, {"_id": self.test_id})
        self.assertEqual(doc["generated_text"], "Hello from real test!")
        self.assertEqual(doc["unit"], "integration")


if __name__ == "__main__":
    unittest.main()
