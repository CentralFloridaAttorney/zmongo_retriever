import unittest
from unittest import mock
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo


class TestZMongoEnv(unittest.TestCase):
    @mock.patch.dict("os.environ", {"MONGO_URI": "", "MONGO_DATABASE_NAME": ""})
    def test_missing_env_variables_raises(self):
        with self.assertRaises(ValueError) as context:
            ZMongo()
        self.assertIn("MONGO_URI and MONGO_DATABASE_NAME must be set", str(context.exception))


if __name__ == "__main__":
    unittest.main()
