import unittest
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoListCollections(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.zmongo = ZMongo()

    async def test_list_collections_success(self):
        # Simulate normal behavior
        self.zmongo.db.list_collection_names = AsyncMock(return_value=["a", "b", "c"])
        collections = await self.zmongo.list_collections()
        self.assertEqual(collections, ["a", "b", "c"])

    async def test_list_collections_exception(self):
        # Simulate an exception being raised
        self.zmongo.db.list_collection_names = AsyncMock(side_effect=RuntimeError("db disconnected"))

        result = await self.zmongo.list_collections()
        self.assertEqual(result, [])  # fallback returns empty list


if __name__ == "__main__":
    unittest.main()
