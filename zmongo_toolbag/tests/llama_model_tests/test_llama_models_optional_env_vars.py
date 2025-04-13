import unittest
from unittest.mock import patch
import os
from zmongo_toolbag.models.llama_model import LlamaModel


class TestLlamaModelOptionalEnvVars(unittest.TestCase):
    @patch.dict(os.environ, {
        "GGUF_MODEL_PATH": "/tmp/test-model.bin",
    })
    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True)
    @patch("zmongo_toolbag.models.llama_model.Llama")
    def test_defaults_applied_when_not_set(self, mock_llama, mock_exists):
        # Unset optional variables if they exist
        os.environ.pop("N_CTX", None)
        os.environ.pop("N_BATCH", None)

        model = LlamaModel()
        self.assertEqual(model.n_ctx, 512)
        self.assertEqual(model.n_batch, 126)
        print("✅ Defaults applied: N_CTX=512, N_BATCH=126")

    @patch.dict(os.environ, {
        "GGUF_MODEL_PATH": "/tmp/test-model.bin",
        "N_CTX": "2048",
        "N_BATCH": "64"
    })
    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True)
    @patch("zmongo_toolbag.models.llama_model.Llama")
    def test_custom_env_values_used(self, mock_llama, mock_exists):
        model = LlamaModel()
        self.assertEqual(model.n_ctx, 2048)
        self.assertEqual(model.n_batch, 64)
        print("✅ Custom values used: N_CTX=2048, N_BATCH=64")

    @patch.dict(os.environ, {
        "GGUF_MODEL_PATH": "/tmp/test-model.bin",
        "N_CTX": "not_an_int",
        "N_BATCH": "also_bad"
    })
    @patch("zmongo_toolbag.models.llama_model.os.path.exists", return_value=True)
    @patch("zmongo_toolbag.models.llama_model.Llama")
    def test_invalid_env_values_fallback_to_defaults(self, mock_llama, mock_exists):
        model = LlamaModel()
        self.assertEqual(model.n_ctx, 512)
        self.assertEqual(model.n_batch, 126)
        print("✅ Invalid values fell back to defaults: N_CTX=512, N_BATCH=126")


if __name__ == "__main__":
    unittest.main()
