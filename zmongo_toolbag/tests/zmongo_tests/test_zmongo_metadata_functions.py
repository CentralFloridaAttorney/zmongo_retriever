import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoMetadataFunctions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()

    async def test_list_collections_success_and_error(self):
        # Success case
        self.repo.db.list_collection_names = AsyncMock(return_value=["users", "cases"])
        result = await self.repo.list_collections()
        self.assertEqual(result, ["users", "cases"])

        # Error case
        self.repo.db.list_collection_names = AsyncMock(side_effect=Exception("DB error"))
        with patch("zmongo_toolbag.zmongo.logger") as mock_logger:
            result = await self.repo.list_collections()
            self.assertEqual(result, [])
            mock_logger.error.assert_called()

    async def test_get_field_names_success_and_error(self):
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{"a": 1, "b": 2}, {"b": 3, "c": 4}])
        self.repo.db["test"].find.return_value.limit.return_value = mock_cursor

        fields = await self.repo.get_field_names("test")
        self.assertCountEqual(fields, ["a", "b", "c"])

        # Error case
        self.repo.db["test"].find.side_effect = Exception("find failed")
        with patch("zmongo_toolbag.zmongo.logger") as mock_logger:
            fields = await self.repo.get_field_names("test")
            self.assertEqual(fields, [])
            mock_logger.error.assert_called()

    async def test_sample_documents_success_and_error(self):
        sample_docs = [{"name": "Alice"}, {"name": "Bob"}]
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=sample_docs)
        self.repo.db["users"].find.return_value.limit.return_value = mock_cursor

        self.repo.serialize_document = lambda x: x  # passthrough
        result = await self.repo.sample_documents("users")
        self.assertEqual(result, sample_docs)

        self.repo.db["users"].find.side_effect = Exception("find failed")
        with patch("zmongo_toolbag.zmongo.logger") as mock_logger:
            result = await self.repo.sample_documents("users")
            self.assertEqual(result, [])
            mock_logger.error.assert_called()

    async def test_count_documents_success_and_error(self):
        self.repo.db["test"].estimated_document_count = AsyncMock(return_value=42)
        count = await self.repo.count_documents("test")
        self.assertEqual(count, 42)

        self.repo.db["test"].estimated_document_count.side_effect = Exception("count fail")
        with patch("zmongo_toolbag.zmongo.logger") as mock_logger:
            count = await self.repo.count_documents("test")
            self.assertEqual(count, 0)
            mock_logger.error.assert_called()

    async def test_get_document_by_id_success_and_error(self):
        obj_id = ObjectId()
        mock_doc = {"_id": obj_id, "value": 123}
        self.repo.db["test"].find_one = AsyncMock(return_value=mock_doc)
        self.repo.serialize_document = lambda x: x  # passthrough

        doc = await self.repo.get_document_by_id("test", str(obj_id))
        self.assertEqual(doc["_id"], obj_id)

        self.repo.db["test"].find_one.side_effect = Exception("find_one failed")
        with patch("zmongo_toolbag.zmongo.logger") as mock_logger:
            doc = await self.repo.get_document_by_id("test", str(obj_id))
            self.assertIsNone(doc)
            mock_logger.error.assert_called()

    async def test_text_search_success_and_error(self):
        mock_docs = [{"_id": ObjectId(), "text": "test result"}]
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_docs)
        self.repo.db["docs"].find.return_value.limit.return_value = mock_cursor
        self.repo.serialize_document = lambda x: x  # passthrough

        result = await self.repo.text_search("docs", "test")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "test result")

        self.repo.db["docs"].find.side_effect = Exception("text search failed")
        with patch("zmongo_toolbag.zmongo.logger") as mock_logger:
            result = await self.repo.text_search("docs", "test")
            self.assertEqual(result, [])
            mock_logger.error.assert_called()
