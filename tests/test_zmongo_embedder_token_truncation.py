import unittest
import tiktoken
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from unittest.mock import MagicMock

class TestZMongoEmbedderTokenTruncation(unittest.TestCase):
    def test_truncate_text_to_max_tokens(self):
        mock_repo = MagicMock()
        embedder = ZMongoEmbedder(repository=mock_repo, collection="test")
        embedder.max_tokens = 10  # set small limit for test
        embedder.encoding_name = "cl100k_base"

        # Create a string longer than 10 tokens
        long_text = "This is a long sentence that will definitely exceed ten tokens for testing purposes."

        truncated = embedder._truncate_text_to_max_tokens(long_text)
        encoding = tiktoken.get_encoding(embedder.encoding_name)
        self.assertLessEqual(len(encoding.encode(truncated)), embedder.max_tokens)

if __name__ == "__main__":
    unittest.main()
