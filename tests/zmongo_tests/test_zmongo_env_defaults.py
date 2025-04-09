import os
import unittest
from unittest.mock import patch
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoEnvDefaults(unittest.TestCase):
    def setUp(self):
        # Backup original environment
        self.original_uri = os.environ.get("MONGO_URI")
        self.original_db = os.environ.get("MONGO_DATABASE_NAME")

        # Remove from environment to test fallback
        os.environ.pop("MONGO_URI", None)
        os.environ.pop("MONGO_DATABASE_NAME", None)

    def tearDown(self):
        # Restore environment to original state
        if self.original_uri:
            os.environ["MONGO_URI"] = self.original_uri
        if self.original_db:
            os.environ["MONGO_DATABASE_NAME"] = self.original_db

    @patch("zmongo_toolbag.zmongo.logger")
    def test_fallback_defaults_and_logger_warnings(self, mock_logger):
        zmongo = ZMongo()

        # Check that fallback values were assigned
        self.assertEqual(zmongo.MONGO_URI, "mongodb://127.0.0.1:27017")
        self.assertEqual(zmongo.MONGO_DB_NAME, "documents")

        # Verify that both warnings were logged
        mock_logger.warning.assert_any_call("⚠️  MONGO_URI is not set. Defaulting to 'mongodb://127.0.0.1:27017'")
        mock_logger.warning.assert_any_call("❌ MONGO_DATABASE_NAME is not set. Defaulting to 'documents'")


if __name__ == "__main__":
    unittest.main()
