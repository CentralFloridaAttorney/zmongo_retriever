import unittest
import asyncio
from zmongo_toolbag.zmongo import ZMongo


class TestInsertEmptyDocuments(unittest.IsolatedAsyncioTestCase):

    async def test_returns_zero_when_documents_empty(self):
        repo = ZMongo()
        collection = "test_empty_insert"

        result = await repo.insert_documents(collection, documents=[])
        self.assertEqual(result, {"inserted_count": 0})


if __name__ == "__main__":
    unittest.main()
