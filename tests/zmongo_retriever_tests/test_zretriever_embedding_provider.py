import unittest
from unittest.mock import patch, MagicMock
from zmongo_toolbag.zretriever import ZRetriever
from langchain_openai import OpenAIEmbeddings


class DummyRepo:
    db = mongo_client = None  # only needed for constructor


class TestZRetrieverEmbeddingProvider(unittest.TestCase):
    @patch("zmongo_toolbag.zretriever.logger")
    def test_embedding_provider_ollama_falls_back_to_openai(self, mock_logger):
        retriever = ZRetriever(repository=DummyRepo(), embedding_provider='ollama')

        # Ensure the fallback model is used
        self.assertIsInstance(retriever.embedding_model, OpenAIEmbeddings)

        # Ensure a warning was logged
        mock_logger.warning.assert_called_with(
            "OllamaEmbeddings not yet supported; falling back to OpenAIEmbeddings."
        )


if __name__ == "__main__":
    unittest.main()
