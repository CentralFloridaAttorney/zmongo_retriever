import unittest
from unittest.mock import patch, MagicMock
import os

from zai_toolbag.zllama import LlamaModel


class TestLlamaModel(unittest.TestCase):

    @patch.dict(os.environ, {
        'GGUF_MODEL_PATH': '/home/overlordx/resources/models/dolphin-2.1-mistral-7b.Q4_0.gguf',
        'GGML_MODEL_URL': 'https://example.com/fake-model-url'
    })
    @patch("zai_toolbag.zllama.os.path.exists", return_value=True)
    @patch("zai_toolbag.zllama.urllib.request.urlretrieve")
    @patch("zai_toolbag.zllama.os.makedirs")
    @patch("zai_toolbag.zllama.os.path.isfile", return_value=False)
    @patch("zai_toolbag.zllama.Llama")
    def test_downloads_if_file_missing(self, mock_llama, mock_isfile, mock_makedirs, mock_urlretrieve, mock_exists):
        model = LlamaModel()
        model.download_model()

        mock_isfile.assert_called_once_with(model.model_path)
        mock_makedirs.assert_called_once_with(os.path.dirname(model.model_path), exist_ok=True)
        mock_urlretrieve.assert_called_once_with(model.model_url, model.model_path)

    @patch.dict(os.environ, {
        'GGUF_MODEL_PATH': '/home/overlordx/resources/models/dolphin-2.1-mistral-7b.Q4_0.gguf',
        'GGML_MODEL_URL': 'https://example.com/fake-model-url'
    })
    @patch("zai_toolbag.zllama.os.path.exists", return_value=True)
    @patch("zai_toolbag.zllama.os.path.isfile", return_value=True)
    @patch("zai_toolbag.zllama.Llama")
    def test_skips_download_if_file_exists(self, mock_llama, mock_isfile, mock_exists):
        model = LlamaModel()
        with patch("builtins.print") as mock_print:
            model.download_model()
            mock_print.assert_any_call("Model file already exists.")

    @patch.dict(os.environ, {
        'GGUF_MODEL_PATH': '/home/overlordx/resources/models/dolphin-2.1-mistral-7b.Q4_0.gguf',
        'GGML_MODEL_URL': 'https://example.com/fake-model-url'
    })
    @patch("zai_toolbag.zllama.os.path.exists", return_value=True)
    @patch("zai_toolbag.zllama.Llama")
    def test_load_model_initializes_llm(self, mock_llama, mock_exists):
        model = LlamaModel()
        self.assertIsNotNone(model.llama_model)
        mock_llama.assert_called_with(
            model_path=model.model_path,
            n_ctx=model.n_ctx,
            n_batch=model.n_batch
        )

    @patch.dict(os.environ, {
        'GGUF_MODEL_PATH': '/home/overlordx/resources/models/dolphin-2.1-mistral-7b.Q4_0.gguf',
        'GGML_MODEL_URL': 'https://example.com/fake-model-url'
    })
    @patch("zai_toolbag.zllama.os.path.exists", return_value=True)
    @patch("zai_toolbag.zllama.Llama")
    def test_generate_prompt_from_template(self, mock_llama, mock_exists):
        model = LlamaModel()
        user_input = "Tell me a joke."
        prompt = model.generate_prompt_from_template(user_input)
        self.assertIn("Tell me a joke.", prompt)
        self.assertTrue(prompt.startswith("<|im_start|>system"))

    @patch.dict(os.environ, {
        'GGUF_MODEL_PATH': '/home/overlordx/resources/models/dolphin-2.1-mistral-7b.Q4_0.gguf',
        'GGML_MODEL_URL': 'https://example.com/fake-model-url'
    })
    @patch("zai_toolbag.zllama.os.path.exists", return_value=True)
    @patch("zai_toolbag.zllama.Llama")
    def test_generate_text_returns_expected_output(self, mock_llama_class, mock_exists):
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
