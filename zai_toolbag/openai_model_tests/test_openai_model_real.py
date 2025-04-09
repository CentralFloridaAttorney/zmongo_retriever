import unittest
from unittest.mock import patch

from bson import ObjectId
from dotenv import load_dotenv

from zai_toolbag.openai_model import OpenAIModel
from zmongo_toolbag.zmongo import ZMongo

load_dotenv()


class TestOpenAIModelReal(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = OpenAIModel()
        cls.zmongo = ZMongo()
        cls.test_collection = "test_openai_results"

    @classmethod
    def tearDownClass(cls):
        # Properly close Mongo connection
        cls.zmongo.mongo_client.close()

    async def test_generate_instruction(self):
        instruction = "What is ZMongo?"
        result = await self.model.generate_instruction(instruction)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    async def test_generate_summary(self):
        long_text = (
            "ZMongo is a Python utility that wraps Motor, providing simplified access "
            "to common MongoDB operations like find, insert, update, and delete using asyncio."
        )
        result = await self.model.generate_summary(long_text)
        self.assertIsInstance(result, str)

    async def test_generate_question_answer(self):
        context = "ZMongo is used for async MongoDB interactions in Python apps."
        question = "What does ZMongo help with?"
        result = await self.model.generate_question_answer(context, question)
        self.assertIn("async", result.lower())

    async def test_generate_zelement_explanation(self):
        doc = {
            "name": "ZMongo",
            "note": "Async MongoDB wrapper",
            "creator": "BPA"
        }
        result = await self.model.generate_zelement_explanation(doc)
        self.assertIn("ZMongo", result)

    async def test_generate_from_template(self):
        template = "Describe the tool: {tool}."
        variables = {"tool": "ZMongo"}
        result = await self.model.generate_from_template(template, variables)
        self.assertIn("ZMongo", result)

    async def test_save_openai_result(self):
        dummy_id = ObjectId()
        result = await self.model.save_openai_result(
            collection_name=self.test_collection,
            record_id=dummy_id,
            field_name="ai_text",
            generated_text="Sample output",
            extra_fields={"source": "test_save"}
        )
        self.assertTrue(isinstance(result, bool))


    def test_record_id_string_conversion(self):
        model = OpenAIModel()
        sample_id_str = str(ObjectId())

        # simulate internal conversion logic
        if isinstance(sample_id_str, str):
            converted_id = ObjectId(sample_id_str)

        self.assertIsInstance(converted_id, ObjectId)
        self.assertEqual(str(converted_id), sample_id_str)

    async def test_call_openai_chat_exception_handling(self):
        model = OpenAIModel()

        with patch("openai.chat.completions.create", side_effect=Exception("Simulated API failure")):
            result = await model._call_openai_chat([{"role": "user", "content": "test"}])

        self.assertTrue(result.startswith("[OpenAI Error] Simulated API failure"))

    async def test_missing_generated_text_or_field_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            await self.model.save_openai_result(
                collection_name="test",
                record_id=ObjectId(),
                field_name="",
                generated_text="some text"
            )
        with self.assertRaises(ValueError):
            await self.model.save_openai_result(
                collection_name="test",
                record_id=ObjectId(),
                field_name="output",
                generated_text=""
            )

    async def test_record_id_str_converts_to_objectid(self):
        model = OpenAIModel()
        string_id = str(ObjectId())  # Create a valid ObjectId string

        # This won't actually run a Mongo update, we're only testing the type conversion
        async def mock_update_document(collection, query, update_data):
            self.assertIsInstance(query["_id"], ObjectId)
            self.assertEqual(str(query["_id"]), string_id)
            return {"matchedCount": 1}  # Simulate success

        class MockZMongo:
            def __init__(self):
                self.mongo_client = None  # no real connection needed

            async def update_document(self, collection, query, update_data):
                return await mock_update_document(collection, query, update_data)

            def close(self): pass  # no-op for cleanup

        result = await model.save_openai_result(
            collection_name="test_openai_results",
            record_id=string_id,
            field_name="ai_text",
            generated_text="Converted ObjectId test",
            zmongo=MockZMongo()
        )

        self.assertTrue(result)

    async def test_stream_response_returns_joined_text(self):
        prompt = "Tell me a short joke about Python programming."
        messages = [{"role": "user", "content": prompt}]

        response = await self.model._call_openai_chat(messages, stream=True)

        # Check the output is a non-empty string (stream joined)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print("\n[Streamed Response]:", response)

if __name__ == "__main__":
    unittest.main()
