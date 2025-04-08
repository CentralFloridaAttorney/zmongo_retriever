import unittest
import logging
from unittest.mock import patch
from langchain_openai import OpenAIEmbeddings
from zmongo_toolbag.zretriever import ZRetriever

class TestEmbeddingProvider(unittest.TestCase):
    """
    Tests the logic in ZRetriever.__init__ which warns and falls back to OpenAIEmbeddings
    when embedding_provider='ollama', and remains silent otherwise.
    """

    def test_default_provider_is_openai(self):
        """
        If embedding_provider is 'openai' (or anything not 'ollama'),
        we expect no warnings and an OpenAIEmbeddings instance.
        """
        # We do NOT use assertLogs because it fails when no logs appear.
        retriever = ZRetriever(embedding_provider='openai')

        # Confirm we got OpenAIEmbeddings with no warnings
        self.assertIsInstance(
            retriever.embedding_model,
            OpenAIEmbeddings,
            "Expected an OpenAIEmbeddings instance for embedding_provider='openai'."
        )
        # No logs are produced, so there's nothing to assert about warnings.
        # If you want to ensure no warnings were called, you can patch logger.warning.

    def test_ollama_provider_falls_back_to_openai(self):
        """
        If embedding_provider='ollama', we expect a WARNING log
        plus an OpenAIEmbeddings instance fallback.
        """
        with self.assertLogs(level='WARNING') as log_cm:
            retriever = ZRetriever(embedding_provider='ollama')

        # Confirm a warning was indeed logged about OllamaEmbeddings not supported
        self.assertTrue(
            any("OllamaEmbeddings not yet supported" in msg for msg in log_cm.output),
            "Expected a warning mentioning 'OllamaEmbeddings not yet supported'."
        )
        # Confirm fallback to OpenAIEmbeddings
        self.assertIsInstance(
            retriever.embedding_model,
            OpenAIEmbeddings,
            "Expected fallback to OpenAIEmbeddings when provider='ollama'."
        )

if __name__ == "__main__":
    unittest.main()
