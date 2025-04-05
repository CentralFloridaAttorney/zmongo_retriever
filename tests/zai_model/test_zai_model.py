import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from zmongo_retriever.zai_toolbag import ZAIModel


class TestZAIModel(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = ZAIModel(config={"debug": True})

    async def test_run_task_returns_expected_structure(self):
        input_text = "Test input for ZAI."
        result = await self.model.run_task(input_text)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["input"], input_text)
        self.assertEqual(result["summary"], input_text.upper())
        self.assertEqual(result["length"], len(input_text))
        self.assertEqual(result["status"], "success")

    async def test_run_task_with_empty_string(self):
        input_text = ""
        result = await self.model.run_task(input_text)

        self.assertEqual(result["input"], "")
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["length"], 0)
        self.assertEqual(result["status"], "success")

    async def test_run_task_with_special_characters(self):
        input_text = "🔥🚀✨"
        result = await self.model.run_task(input_text)

        self.assertEqual(result["summary"], input_text.upper())  # Unicode .upper() is safe
        self.assertEqual(result["length"], len(input_text))
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
