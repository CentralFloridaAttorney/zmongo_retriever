import unittest
from typing import Optional, Any, Union

from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from models.base_result_saver import BaseResultSaver


class DummySaver(BaseResultSaver):
    """Concrete subclass for testing the pass condition of the abstract method."""

    def __init__(self):
        self.zmongo = ZMongo()

    async def save(
        self,
        collection_name: str,
        record_id: Union[str, ObjectId],
        field_name: str,
        generated_text: str,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> bool:
        if isinstance(record_id, str):
            record_id = ObjectId(record_id)

        update_data = {"$set": {field_name: generated_text}}
        if extra_fields:
            update_data["$set"].update(extra_fields)

        result = await self.zmongo.update_document(
            collection=collection_name,
            query={"_id": record_id},
            update_data=update_data,
            upsert=True
        )
        return result.matched_count > 0 or result.upserted_id is not None


class TestBaseResultSaverPassCondition(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.saver = DummySaver()
        self.collection = "test_base_saver"
        self.test_id = ObjectId()

        # Pre-insert doc
        await self.saver.zmongo.insert_document(self.collection, {
            "_id": self.test_id,
            "starter": True
        })

    async def asyncTearDown(self):
        await self.saver.zmongo.delete_document(self.collection, {"_id": self.test_id})
        await self.saver.zmongo.close()

    async def test_save_updates_field(self):
        success = await self.saver.save(
            collection_name=self.collection,
            record_id=self.test_id,
            field_name="result",
            generated_text="Hello from BaseResultSaver pass test",
            extra_fields={"verified": True}
        )
        self.assertTrue(success)

        doc = await self.saver.zmongo.find_document(self.collection, {"_id": self.test_id})
        self.assertEqual(doc["result"], "Hello from BaseResultSaver pass test")
        self.assertEqual(doc["verified"], True)


if __name__ == "__main__":
    unittest.main()
