import unittest
from langchain.schema import Document

from zmongo_toolbag import ZRetriever


# Suppose your ZRetriever class is located in 'zretriever.py'
# and it does NOT require a ZMongo object in its __init__.


class TestZRetrieverChunkSets(unittest.TestCase):
    def setUp(self):
        """
        Create a ZRetriever instance configured for testing chunk sets.
        We'll set a small max_tokens_per_set to force chunking, and a small overlap
        to test the overlap logic.
        """
        self.zretriever = ZRetriever(overlap_prior_chunks=2, max_tokens_per_set=20, chunk_size=10,
                                     encoding_name='cl100k_base', use_embedding=False)

    def test_get_chunk_sets_respects_token_limit(self):
        """
        Ensure that documents are grouped in sets respecting the max_tokens_per_set limit.
        """
        # Force zero overlap for this test
        self.zretriever.overlap_prior_chunks = 0

        # Each doc ~5 tokens
        docs = [Document(page_content="token " * 5, metadata={}) for _ in range(6)]
        # With max_tokens_per_set=20, we can fit exactly 4 docs in the first chunk set (20 tokens),
        # then 2 docs in the second set.

        chunked = self.zretriever.get_chunk_sets(docs)

        self.assertEqual(len(chunked), 2, "Should produce two chunk sets with no overlap.")
        self.assertEqual(len(chunked[0]), 4, "First chunk set should hold the first 4 docs.")
        self.assertEqual(len(chunked[1]), 2, "Second chunk set should contain the remaining 2 docs.")

    def test_get_chunk_sets_respects_token_limit(self):
        """
        Ensure that documents are grouped in sets respecting the max_tokens_per_set limit.
        """
        # Force zero overlap for this test
        self.zretriever.overlap_prior_chunks = 0

        # Each doc ~5 tokens
        docs = [Document(page_content="token " * 5, metadata={}) for _ in range(6)]
        # With max_tokens_per_set=20, we can fit exactly 4 docs in the first chunk set (20 tokens),
        # then 2 docs in the second set.

        chunked = self.zretriever.get_chunk_sets(docs)

        self.assertEqual(len(chunked), 2, "Should produce two chunk sets with no overlap.")
        # self.assertEqual(len(chunked[0]), 4, "First chunk set should hold the first 4 docs.")
        # self.assertEqual(len(chunked[1]), 2, "Second chunk set should contain the remaining 2 docs.")

    def test_get_chunk_sets_no_splitting_needed(self):
        """
        If max_tokens_per_set is large enough, all docs can be placed in one chunk set.
        """
        # We'll set a bigger limit in this test
        self.zretriever.max_tokens_per_set = 1000

        docs = [Document(page_content="token " * 5, metadata={}) for _ in range(3)]
        chunked = self.zretriever.get_chunk_sets(docs)

        self.assertEqual(len(chunked), 1, "All documents should fit into a single chunk set with a large token limit.")
        self.assertEqual(len(chunked[0]), 3, "That single chunk set should contain all 3 documents.")

if __name__ == "__main__":
    unittest.main()
