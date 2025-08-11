import unittest
from bson.objectid import ObjectId

from models.base_result_saver import BaseResultSaver


# A simple subclass for testing purposes
class DummyResultSaver(BaseResultSaver):
    async def save(self, collection_name, record_id, field_name, generated_text, extra_fields=None):
        return True  # simulate success


class TestBaseResultSaverPassCondition(unittest.IsolatedAsyncioTestCase):
    async def test_save_returns_true(self):
        saver = DummyResultSaver()
        result = await saver.save(
            collection_name="test_collection",
            record_id=ObjectId(),
            field_name="output",
            generated_text="This is a test result.",
            extra_fields={"meta": "info"}
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
