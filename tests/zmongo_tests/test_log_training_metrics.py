import unittest
from unittest.mock import patch, MagicMock
from zmongo_toolbag.zmongo import ZMongo


class TestLogTrainingMetrics(unittest.TestCase):
    @patch("zmongo_toolbag.zmongo.logger")
    def test_log_training_metrics_inserts_doc_and_logs(self, mock_logger):
        """
        Verifies that log_training_metrics logs and inserts when successful.
        """
        zmongo = ZMongo()
        mock_training_collection = MagicMock()
        zmongo.sync_db = {"training_metrics": mock_training_collection}

        sample_metrics = {"loss": 0.123, "accuracy": 0.99}
        zmongo.log_training_metrics(sample_metrics)

        self.assertTrue(mock_training_collection.insert_one.called)
        self.assertTrue(mock_logger.info.called)
        self.assertIn("Logged training metrics", mock_logger.info.call_args[0][0])

    @patch("zmongo_toolbag.zmongo.logger")
    def test_log_training_metrics_handles_insert_failure(self, mock_logger):
        """
        Verifies that if insert_one fails, the error is caught and logged.
        """
        zmongo = ZMongo()

        # Simulate a failure when inserting
        mock_training_collection = MagicMock()
        mock_training_collection.insert_one.side_effect = RuntimeError("Insert failed")
        zmongo.sync_db = {"training_metrics": mock_training_collection}

        sample_metrics = {"loss": 0.555, "accuracy": 0.777}
        zmongo.log_training_metrics(sample_metrics)

        # Make sure error logging was triggered
        self.assertTrue(mock_logger.error.called)
        logged_error = mock_logger.error.call_args[0][0]
        self.assertIn("Failed to log training metrics:", logged_error)
        self.assertIn("Insert failed", logged_error)


if __name__ == "__main__":
    unittest.main()
