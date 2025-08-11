import unittest
from bson.objectid import ObjectId

from zmongo_toolbag import OpenAIResultSaver


class TestOpenAIResultSaverInputValidation(unittest.IsolatedAsyncioTestCase):
    async def test_raises_if_field_name_missing(self):
        saver = OpenAIResultSaver()
        with self.assertRaises(ValueError) as ctx:
            await saver.save(
                collection_name="test_collection",
                record_id=ObjectId(),
                field_name="",  # <-- Invalid
                generated_text="Valid Text"
            )
        self.assertIn("field name must be provided", str(ctx.exception))

    async def test_raises_if_generated_text_missing(self):
        saver = OpenAIResultSaver()
        with self.assertRaises(ValueError) as ctx:
            await saver.save(
                collection_name="test_collection",
                record_id=ObjectId(),
                field_name="field",
                generated_text=""  # <-- Invalid
            )
        self.assertIn("Generated text and field name must be provided", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
