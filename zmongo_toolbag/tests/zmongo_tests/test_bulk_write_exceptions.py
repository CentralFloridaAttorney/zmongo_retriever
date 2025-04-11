import unittest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import BulkWriteError, PyMongoError
from pymongo import InsertOne
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoBulkWriteExceptions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.zmongo = ZMongo()
        self.collection_name = "test_bulk_write"
        self.valid_operations = [InsertOne({"name": "test"})]

    async def test_bulk_write_bulk_write_error(self):
        # Create mocked collection with a bulk_write that raises BulkWriteError
        mock_collection = AsyncMock()
        mock_collection.bulk_write.side_effect = BulkWriteError({"writeErrors": [{"msg": "fail"}]})

        # Patch the .db attribute to return our mocked collection
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        self.zmongo.db = mock_db

        result = await self.zmongo.bulk_write(self.collection_name, self.valid_operations)
        self.assertIn("error", result)
        self.assertIn("writeErrors", result["error"])

    async def test_bulk_write_pymongo_error(self):
        mock_collection = AsyncMock()
        mock_collection.bulk_write.side_effect = PyMongoError("some pymongo error")

        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        self.zmongo.db = mock_db

        result = await self.zmongo.bulk_write(self.collection_name, self.valid_operations)
        self.assertIn("error", result)
        self.assertIn("pymongo error", result["error"].lower())

    async def test_bulk_write_generic_exception(self):
        mock_collection = AsyncMock()
        mock_collection.bulk_write.side_effect = Exception("unexpected error")

        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        self.zmongo.db = mock_db

        result = await self.zmongo.bulk_write(self.collection_name, self.valid_operations)
        self.assertIn("error", result)
        self.assertIn("unexpected error", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
