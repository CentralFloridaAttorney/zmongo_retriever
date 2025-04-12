import unittest
from unittest.mock import patch, MagicMock
import os

from zmongo_toolbag.models.llama_model import LlamaModel


@patch.dict(os.environ, {
    "GGUF_MODEL_PATH": "/tmp/test-model.bin",
    "N_CTX": "2048",
    "N_BATCH": "128",
    "DEFAULT_MAX_TOKENS": "100",
    "DEFAULT_TEMPERATURE": "0.8",
    "DEFAULT_TOP_P": "0.9",
    "DEFAULT_STOP": "###"
})
class TestLlamaModel(unittest.TestCase):

    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True)
    @patch("zmongo_toolbag.models.llama_model.Llama")
    def test_init_and_load_model_success(self, mock_llama, mock_exists):
        model = LlamaModel()
        mock_llama.assert_called_once_with(
            model_path="/tmp/test-model.bin",
            n_ctx=2048,
            n_batch=128
        )
        self.assertIsNotNone(model.llm)
        print("✅ Model loaded successfully with mocked Llama.")

    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=False)
    def test_model_file_missing_raises_error(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            LlamaModel()
        print("✅ Correctly raised FileNotFoundError when model file is missing.")

    def test_generate_prompt_from_template_includes_user_input(self):
        with patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True), \
             patch("zmongo_toolbag.models.llama_model.Llama"):
            model = LlamaModel()
            user_input = "What is the capital of France?"
            prompt = model.generate_prompt_from_template(user_input)
            print("📥 Generated Prompt:\n", prompt)
            self.assertIn("What is the capital of France?", prompt)
            self.assertTrue(prompt.startswith("<|im_start|>system"))

    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True)
    @patch("zmongo_toolbag.models.llama_model.Llama")
    def test_generate_text_calls_llm_and_returns_trimmed_text(self, mock_llama_class, mock_exists):
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": "\n This is a response. \n"}]
        }
        mock_llama_class.return_value = mock_llm_instance

        model = LlamaModel()
        model.llm = mock_llm_instance

        result = model.generate_text("Hello?")
        print("💬 Generated Text:", repr(result))
        self.assertEqual(result, "This is a response.")

    def test_missing_env_variable_raises_value_error(self):
        with patch.dict(os.environ, {"GGUF_MODEL_PATH": ""}):
            with self.assertRaises(ValueError):
                LlamaModel()
        print("✅ Correctly raised ValueError for missing GGUF_MODEL_PATH.")

    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True)
    @patch("zmongo_toolbag.models.llama_model.Llama", side_effect=Exception("LLM failure"))
    def test_load_model_fails_gracefully(self, mock_llama, mock_exists):
        with self.assertRaises(Exception) as context:
            LlamaModel()
        print("❌ LLM Load Exception Caught:", str(context.exception))
        self.assertIn("LLM failure", str(context.exception))


if __name__ == "__main__":
    unittest.main()
