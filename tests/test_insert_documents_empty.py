import unittest
import asyncio
from unittest.mock import MagicMock

from zmongo_retriever.zmongo_toolbag import ZMongo


class TestInsertDocumentsEarlyReturn(unittest.IsolatedAsyncioTestCase):
    async def test_insert_documents_empty_list_returns_zero(self):
        # Arrange
        repo = ZMongo()
        repo.db = MagicMock()  # avoid actual MongoDB calls

        # Act
        result = await repo.insert_documents(collection="test", documents=[])

        # Assert
        self.assertEqual(result, {"inserted_count": 0})


if __name__ == "__main__":
    asyncio.run(unittest.main())
