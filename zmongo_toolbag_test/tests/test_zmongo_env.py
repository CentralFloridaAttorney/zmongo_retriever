import os
import unittest
from unittest import mock
from zmongo_toolbag.zmongo import ZMongo

class TestZMongoEnv(unittest.TestCase):

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_default_env_values_used_when_missing(self):
        zmongo = ZMongo()
        self.assertEqual(zmongo.MONGO_URI, "mongodb://127.0.0.1:27017")
        self.assertEqual(zmongo.MONGO_DB_NAME, "documents")

if __name__ == "__main__":
    unittest.main()
