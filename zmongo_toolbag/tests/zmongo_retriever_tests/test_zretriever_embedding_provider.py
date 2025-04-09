import unittest
from unittest.mock import patch, MagicMock
from langchain_openai import OpenAIEmbeddings

from zmongo_toolbag import ZRetriever


class TestZRetrieverEmbeddingProvider(unittest.TestCase):
    @patch("zmongo_toolbag.zretriever.logger")
    @patch("zmongo_toolbag.zretriever.OpenAIEmbeddings")
    def test_fallback_to_openai_when_ollama_selected(self, mock_openai, mock_logger):
        # Arrange
        mock_repo = MagicMock()

        # Act
        retriever = ZRetriever(repository=mock_repo, embedding_provider='ollama')

        # Assert
        mock_logger.warning.assert_called_once_with(
            "OllamaEmbeddings not yet supported; falling back to OpenAIEmbeddings."
        )
        mock_openai.assert_called()
        self.assertIsInstance(retriever.embedding_model, mock_openai.return_value.__class__)
