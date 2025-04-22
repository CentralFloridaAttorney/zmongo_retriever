import unittest
from unittest.mock import AsyncMock
from langchain.schema import Document

from zmongo_toolbag.zretriever import ZRetriever

class TestInvokeChunkSets(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_returns_chunk_sets_when_max_tokens_per_set_positive(self):
        """
        Ensures that if max_tokens_per_set >= 1, the invoke method returns
        chunk sets instead of raw documents.
        """
        # 1. Initialize a ZRetriever with a positive max_tokens_per_set
        retriever = ZRetriever(overlap_prior_chunks=2, max_tokens_per_set=20, chunk_size=10)

        # 2. Mock out get_zdocuments so we control the returned documents
        fake_docs = [
            Document(page_content="token " * 5, metadata={"doc_index": 0}),
            Document(page_content="token " * 5, metadata={"doc_index": 1}),
            Document(page_content="token " * 5, metadata={"doc_index": 2}),
        ]
        retriever.get_zdocuments = AsyncMock(return_value=fake_docs)

        # 3. Since invoke is async, we must await it to get the actual result
        result = await retriever.invoke(
            collection="test_collection",
            object_ids=["dummy_id"]
        )

        # 4. Verify the returned result is a list of chunk sets
        self.assertIsInstance(result, list, "Expected a list of chunk sets (list of lists).")
        self.assertTrue(all(isinstance(chunk_list, list) for chunk_list in result),
                        "Each chunk set should be a list of Document objects.")

        # 5. Because each doc is ~5 tokens and max_tokens_per_set=20,
        # all 3 might fit in one chunk set
        self.assertEqual(len(result), 1, "Expected exactly one chunk set if all docs fit.")
        self.assertEqual(len(result[0]), 3, "All docs should be in that one chunk set.")

        # 6. Verify get_zdocuments was called with the expected arguments
        retriever.get_zdocuments.assert_called_once_with(
            "test_collection",
            ["dummy_id"],
            "casebody.data.opinions.0.text",
            None
        )

if __name__ == "__main__":
    unittest.main()
