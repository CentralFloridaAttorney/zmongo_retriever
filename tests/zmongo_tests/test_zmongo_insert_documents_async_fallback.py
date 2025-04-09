import asyncio
import unittest
from unittest.mock import MagicMock, patch
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoInsertDocumentsSyncFallback(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()

    @patch("asyncio.get_running_loop")
    async def test_insert_documents_uses_sync(self, mock_get_loop):
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        # Create a Future and set a return value
        fut = asyncio.Future()
        fut.set_result({"inserted_count": 3})
        mock_loop.run_in_executor.return_value = fut

        collection = "test"
        documents = [{"x": 1}, {"x": 2}, {"x": 3}]

        result = await self.repo.insert_documents(collection, documents, use_sync=True)

        mock_get_loop.assert_called_once()
        mock_loop.run_in_executor.assert_called_once_with(
            None, self.repo.insert_documents_sync, collection, documents, 1000
        )
        self.assertEqual(result, {"inserted_count": 3})
