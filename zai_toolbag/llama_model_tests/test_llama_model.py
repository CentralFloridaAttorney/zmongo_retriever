import unittest
from unittest.mock import patch, MagicMock
import os

from zai_toolbag.zllama import LlamaModel


class TestLlamaModel(unittest.TestCase):

    # Clear or mock environment variables for testing
    @patch.dict(os.environ, {'MODEL_PATH': ''})  # Or provide a mock valid path for testing
    @patch("zmongo_toolbag.llama_model.urllib.request.urlretrieve")
    @patch("zmongo_toolbag.llama_model.os.makedirs")
    @patch("zmongo_toolbag.llama_model.os.path.isfile", return_value=False)
    @patch("zmongo_toolbag.llama_model.Llama")
    def test_downloads_if_file_missing(self, mock_llama, mock_isfile, mock_makedirs, mock_urlretrieve):
        model = LlamaModel()
        model.model_path = "/tmp/test-model.bin"
        model.model_url = "https://example.com/fake-model-url"

        model.download_model()

        mock_isfile.assert_called_once_with(model.model_path)
        mock_makedirs.assert_called_once_with(os.path.dirname(model.model_path), exist_ok=True)
        mock_urlretrieve.assert_called_once_with(model.model_url, model.model_path)

    @patch.dict(os.environ, {'MODEL_PATH': ''})  # Or provide a mock valid path for testing
    @patch("zmongo_toolbag.llama_model.os.path.isfile", return_value=True)
    @patch("zmongo_toolbag.llama_model.Llama")
    def test_skips_download_if_file_exists(self, mock_llama, mock_isfile):
        model = LlamaModel()
        model.model_path = "/tmp/test-model.bin"

        with patch("builtins.print") as mock_print:
            model.download_model()
            mock_print.assert_called_with("Model file already exists.")

    @patch.dict(os.environ, {'MODEL_PATH': ''})  # Or provide a mock valid path for testing
    @patch("zmongo_toolbag.llama_model.Llama")
    def test_load_model_initializes_llm(self, mock_llama):
        model = LlamaModel()
        model.model_path = "/tmp/fake-path"
        model.load_model()

        # We expect Llama to have been called twice: once in __init__, once in load_model
        self.assertGreaterEqual(mock_llama.call_count, 2)
        mock_llama.assert_any_call(
            model_path=model.model_path,
            n_ctx=model.n_ctx,
            n_batch=model.n_batch
        )

    @patch.dict(os.environ, {'MODEL_PATH': ''})  # Or provide a mock valid path for testing
    @patch("zmongo_toolbag.llama_model.Llama")
    def test_generate_prompt_from_template(self, mock_llama):
        model = LlamaModel()
        user_input = "Tell me a joke."
        prompt = model.generate_prompt_from_template(user_input)
        self.assertIn("Tell me a joke.", prompt)
        self.assertTrue(prompt.startswith("<|im_start|>system"))

    @patch.dict(os.environ, {'MODEL_PATH': ''})  # Or provide a mock valid path for testing
    @patch("zmongo_toolbag.llama_model.Llama")
    def test_generate_text_returns_expected_output(self, mock_llama_class):
        mock_llama_instance = MagicMock()
        mock_llama_instance.return_value = {
            "choices": [{"text": "\nHere is a joke."}]
        }
        mock_llama_class.return_value = mock_llama_instance

        model = LlamaModel()
        model.llama_model = mock_llama_instance

        result = model.generate_text(prompt="Tell me a joke.")
        self.assertEqual(result, "Here is a joke.")


if __name__ == "__main__":
    unittest.main()
