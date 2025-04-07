import unittest
from unittest.mock import patch, MagicMock
import os

from zmongo_toolbag.llama_model import LlamaModel


class TestLlamaModelDownload(unittest.TestCase):
    @patch("zmongo_toolbag.llama_model.urllib.request.urlretrieve")
    @patch("zmongo_toolbag.llama_model.os.makedirs")
    @patch("zmongo_toolbag.llama_model.os.path.isfile", return_value=False)
    def test_downloads_if_file_missing(self, mock_isfile, mock_makedirs, mock_urlretrieve):
        model = LlamaModel()
        model.model_path = "/tmp/test-model.bin"
        model.model_url = "https://example.com/fake-model-url"

        model.download_model()

        mock_isfile.assert_called_once_with(model.model_path)
        mock_makedirs.assert_called_once_with(os.path.dirname(model.model_path), exist_ok=True)
        mock_urlretrieve.assert_called_once_with(model.model_url, model.model_path)

    @patch("zmongo_toolbag.llama_model.os.path.isfile", return_value=True)
    def test_skips_download_if_file_exists(self, mock_isfile):
        model = LlamaModel()
        model.model_path = "/tmp/test-model.bin"

        with patch("builtins.print") as mock_print:
            model.download_model()
            mock_print.assert_called_with("Model file already exists.")


if __name__ == "__main__":
    unittest.main()
