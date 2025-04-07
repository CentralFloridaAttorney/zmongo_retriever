import unittest
from unittest.mock import MagicMock, patch
from zmongo_toolbag.zmongo import ZMongo
from datetime import datetime
import logging


class TestZMongo(unittest.TestCase):
    def setUp(self):
        self.zmongo = ZMongo()
        self.zmongo.sync_db = {"training_metrics": MagicMock()}
        self.zmongo.sync_db["training_metrics"].insert_one = MagicMock()

    @patch("zmongo_toolbag.zmongo.logger")
    def test_log_training_metrics_success(self, mock_logger):
        metrics = {"accuracy": 0.95, "loss": 0.05}
        self.zmongo.log_training_metrics(metrics)

        # Check insert_one called
        self.zmongo.sync_db["training_metrics"].insert_one.assert_called_once()
        inserted_doc = self.zmongo.sync_db["training_metrics"].insert_one.call_args[0][0]
        self.assertIn("timestamp", inserted_doc)
        self.assertAlmostEqual(inserted_doc["accuracy"], 0.95)
        self.assertAlmostEqual(inserted_doc["loss"], 0.05)
        mock_logger.info.assert_called()

    @patch("zmongo_toolbag.zmongo.logger")
    def test_log_training_metrics_failure(self, mock_logger):
        self.zmongo.sync_db["training_metrics"].insert_one.side_effect = Exception("Insert failed")
        metrics = {"accuracy": 0.99}
        self.zmongo.log_training_metrics(metrics)

        mock_logger.error.assert_called_once()
