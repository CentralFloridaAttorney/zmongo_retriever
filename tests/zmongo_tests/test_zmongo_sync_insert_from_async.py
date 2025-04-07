import unittest
from unittest.mock import MagicMock, patch
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoInsertDocumentsSync(unittest.TestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.collection = "test_collection"

        # Patch sync_db
        self.repo.sync_db = MagicMock()

    def test_insert_documents_sync_success(self):
        documents = [{"val": 1}, {"val": 2}]
        fake_ids = [ObjectId(), ObjectId()]

        # Mock insert_many to return object with inserted_ids
        self.repo.sync_db[self.collection].insert_many.return_value.inserted_ids = fake_ids

        result = self.repo.insert_documents_sync(self.collection, documents)
        self.assertEqual(result["inserted_count"], 2)
        self.assertNotIn("errors", result)

    def test_insert_documents_sync_empty_list(self):
        result = self.repo.insert_documents_sync(self.collection, [])
        self.assertEqual(result["inserted_count"], 0)
        self.assertNotIn("errors", result)

    def test_insert_documents_sync_partial_failure(self):
        documents = [{"val": i} for i in range(3)]

        # Simulate exception on insert
        def insert_many_raises(batch, ordered):
            raise Exception("Mock insertion failed")

        self.repo.sync_db[self.collection].insert_many.side_effect = insert_many_raises

        result = self.repo.insert_documents_sync(self.collection, documents, batch_size=2)
        self.assertEqual(result["inserted_count"], 0)
        self.assertIn("errors", result)
        self.assertTrue(any("Mock insertion failed" in e for e in result["errors"]))
