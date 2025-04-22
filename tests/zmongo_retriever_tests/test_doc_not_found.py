import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zretriever import ZRetriever


class TestDocNotFound(unittest.IsolatedAsyncioTestCase):
    """
    Tests that ZRetriever.get_zdocuments logs a warning and returns an empty list
    when no document is found for a given ID.
    """

    @patch("zmongo_toolbag.zretriever.logger")
    async def test_warns_and_skips_if_no_doc_found(self, mock_logger):
        # Setup mock repository
        mock_repo = MagicMock()
        mock_repo.find_document = AsyncMock(return_value=None)
        mock_repo.db = MagicMock()
        mock_repo.mongo_client = MagicMock()

        # Inject mock repository correctly
        retriever = ZRetriever()

        # Run test
        test_id = str(ObjectId())
        docs = await retriever.get_zdocuments("test_collection", [test_id])

        self.assertEqual(docs, [], "Expected an empty list if no doc is found.")
        self.assertTrue(mock_logger.warning.called, "Expected logger.warning to be called.")

        warning_calls = mock_logger.warning.call_args_list
        found_expected = any("No document found for ID:" in str(call[0][0]) for call in warning_calls)

        self.assertTrue(found_expected, "Expected a warning containing 'No document found for ID:'")


if __name__ == "__main__":
    unittest.main()
