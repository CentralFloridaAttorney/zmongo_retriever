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
    @patch("zmongo_toolbag.zretriever.ZMongo", autospec=True)
    async def test_warns_and_skips_if_no_doc_found(self, mock_zmongo_class, mock_logger):
        # Get the mock instance
        mock_zmongo_instance = mock_zmongo_class.return_value

        # Manually add the 'db' attribute and 'mongo_client' to avoid attribute errors
        mock_zmongo_instance.db = MagicMock()
        mock_zmongo_instance.mongo_client = MagicMock()

        # Make find_document return None to simulate "no document found"
        mock_zmongo_instance.find_document = AsyncMock(return_value=None)

        # Create the ZRetriever (which will use the patched ZMongo)
        retriever = ZRetriever()

        # Call get_zdocuments with a random ObjectId
        docs = await retriever.get_zdocuments(
            "test_collection",
            [str(ObjectId())]
        )

        # Should return an empty list if no doc is found
        self.assertEqual(docs, [], "Expected an empty list if no doc is found.")

        # Now verify that a warning was logged
        # We check the actual log messages that were called
        self.assertTrue(mock_logger.warning.called, "logger.warning should have been called at least once.")

        # Get all calls to logger.warning
        warning_calls = mock_logger.warning.call_args_list
        # We'll check if any of them match the expected message
        found_expected_warning = any(
            "No document found for ID:" in str(call[0][0])  # call[0][0] is the first positional arg to logger.warning
            for call in warning_calls
        )

        self.assertTrue(
            found_expected_warning,
            "Expected a warning containing 'No document found for ID:'"
        )

if __name__ == "__main__":
    unittest.main()
