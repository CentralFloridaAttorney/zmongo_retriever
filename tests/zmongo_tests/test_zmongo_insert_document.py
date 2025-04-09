import unittest
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoInsertDocument(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()

    async def test_insert_document_exception(self):
        collection = "test_collection"
        document = {"key": "value"}

        # Simulate insert_one raising an exception
        self.repo.db[collection].insert_one = AsyncMock(side_effect=Exception("insert failed"))

        result = await self.repo.insert_document(collection, document)

        self.assertIsNone(result)
