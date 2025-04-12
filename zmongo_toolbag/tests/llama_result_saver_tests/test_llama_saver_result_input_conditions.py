import unittest
import asyncio
from bson import ObjectId

from zmongo_toolbag import ZMongo
from zmongo_toolbag.models.llama_result_saver import LlamaResultSaver


class TestLlamaResultSaverInputConditions(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.collection = "test_llama_conditions"
        self.test_id = ObjectId()
        self.test_id_str = str(self.test_id)
        self.test_text = "Some text"
        self.test_field = "response"

    async def test_raises_if_field_name_missing(self):
        saver = LlamaResultSaver()
        with self.assertRaises(ValueError):
            await saver.save(
                collection_name=self.collection,
                record_id=self.test_id,
                field_name="",
                generated_text=self.test_text
            )

    async def test_raises_if_generated_text_missing(self):
        saver = LlamaResultSaver()
        with self.assertRaises(ValueError):
            await saver.save(
                collection_name=self.collection,
                record_id=self.test_id,
                field_name=self.test_field,
                generated_text=""
            )

    async def test_str_record_id_gets_converted(self):
        saver = LlamaResultSaver()
        inserted = await saver.zmongo.insert_document(self.collection, {
            "_id": self.test_id,
            "name": "convert test"
        })
        self.assertIsNotNone(inserted)

        result = await saver.save(
            collection_name=self.collection,
            record_id=self.test_id_str,  # <-- str type
            field_name="check",
            generated_text="converted!"
        )
        self.assertTrue(result)

        # Use a new ZMongo instance for verification and cleanup
        fresh_zmongo = ZMongo()

        doc = await fresh_zmongo.find_document(self.collection, {"_id": self.test_id})
        self.assertEqual(doc["check"], "converted!")

        await fresh_zmongo.delete_document(self.collection, {"_id": self.test_id})
        await fresh_zmongo.close()

    async def test_client_closes_when_should_close_true(self):
        saver = LlamaResultSaver()  # no zmongo passed = _should_close = True
        self.assertTrue(saver._should_close)

        # Close is triggered after this call
        await saver.save(
            collection_name=self.collection,
            record_id=self.test_id,
            field_name=self.test_field,
            generated_text=self.test_text
        )

        # Verify it's closed (client is disconnected)
        self.assertTrue(saver.zmongo.mongo_client is not None)
        await saver.zmongo.close()  # cleanup just in case


if __name__ == "__main__":
    unittest.main()
