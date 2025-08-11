import unittest
from bson.objectid import ObjectId

from zai_toolbag.openai_model import OpenAIModel
from zmongo_toolbag import ZMongo


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


if __name__ == "__main__":
    unittest.main()
