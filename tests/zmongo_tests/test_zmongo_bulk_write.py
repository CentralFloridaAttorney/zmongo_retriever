import unittest
from bson import ObjectId
from pymongo import InsertOne
from pymongo.errors import BulkWriteError, PyMongoError
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoBulkWrite(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.zmongo = ZMongo()
        self.collection = "bulk_write_test"

    async def test_bulk_write_bulk_write_error_mocked(self):
        # ✅ Fully control the collection returned by self.db[collection]
        mock_collection = MagicMock()
        mock_collection.bulk_write = AsyncMock(side_effect=BulkWriteError({"writeErrors": [{"errmsg": "bad op"}]}))
        self.zmongo.db = MagicMock()
        self.zmongo.db.__getitem__.return_value = mock_collection

        result = await self.zmongo.bulk_write(self.collection, [InsertOne({"x": 1})])

        self.assertIn("error", result)
        self.assertIn("writeErrors", result["error"])

    async def test_bulk_write_pymongo_error_mocked(self):
        mock_collection = MagicMock()
        mock_collection.bulk_write = AsyncMock(side_effect=PyMongoError("connection lost"))
        self.zmongo.db = MagicMock()
        self.zmongo.db.__getitem__.return_value = mock_collection

        result = await self.zmongo.bulk_write(self.collection, [InsertOne({"x": 1})])

        self.assertIn("error", result)
        self.assertIn("connection lost", result["error"])

    async def test_bulk_write_unexpected_error_mocked(self):
        mock_collection = MagicMock()
        mock_collection.bulk_write = AsyncMock(side_effect=RuntimeError("unexpected fail"))
        self.zmongo.db = MagicMock()
        self.zmongo.db.__getitem__.return_value = mock_collection

        result = await self.zmongo.bulk_write(self.collection, [InsertOne({"x": 1})])

        self.assertIn("error", result)
        self.assertIn("unexpected fail", result["error"])
