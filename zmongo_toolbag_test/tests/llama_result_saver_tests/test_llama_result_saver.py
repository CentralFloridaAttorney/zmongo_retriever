import unittest
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from models import LlamaResultSaver


class TestLlamaResultSaverIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.zmongo = ZMongo()
        self.saver = LlamaResultSaver(zmongo=self.zmongo)
        self.collection = "test_llama_results"
        self.test_id = ObjectId()
        self.test_field = "generated"
        self.test_text = "This is a test from LlamaResultSaver."
        self.test_metadata = {"source": "test_suite"}

        # Insert a dummy document to update
        await self.zmongo.insert_document(self.collection, {
            "_id": self.test_id,
            "placeholder": True
        })

    async def asyncTearDown(self):
        await self.zmongo.delete_document(self.collection, {"_id": self.test_id})
        await self.zmongo.close()

    async def test_real_save_updates_field(self):
        result = await self.saver.save(
            collection_name=self.collection,
            record_id=self.test_id,
            field_name=self.test_field,
            generated_text=self.test_text,
            extra_fields=self.test_metadata
        )

        self.assertTrue(result, "Expected the save method to return True")

        # Confirm it was updated in the database
        doc = await self.zmongo.find_document(self.collection, {"_id": self.test_id})
        self.assertEqual(doc[self.test_field], self.test_text)
        self.assertEqual(doc["source"], "test_suite")


if __name__ == "__main__":
    unittest.main()
