import unittest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoMetadataFunctions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()

    async def test_list_collections(self):
        self.repo.db.list_collection_names = AsyncMock(return_value=["collection1", "collection2"])
        result = await self.repo.list_collections()
        self.assertEqual(result, ["collection1", "collection2"])

    async def test_get_field_names(self):
        sample_docs = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
        cursor_mock = MagicMock()
        cursor_mock.to_list = AsyncMock(return_value=sample_docs)
        self.repo.db["test"].find.return_value.limit.return_value = cursor_mock

        fields = await self.repo.get_field_names("test")
        self.assertCountEqual(fields, ["a", "b", "c"])

    async def test_sample_documents(self):
        sample_docs = [{"x": 1}, {"y": 2}]
        cursor_mock = MagicMock()
        cursor_mock.to_list = AsyncMock(return_value=sample_docs)
        self.repo.db["test"].find.return_value.limit.return_value = cursor_mock

        self.repo.serialize_document = lambda doc: doc  # skip serialization
        result = await self.repo.sample_documents("test")
        self.assertEqual(result, sample_docs)

    async def test_count_documents(self):
        self.repo.db["test"].estimated_document_count = AsyncMock(return_value=42)
        count = await self.repo.count_documents("test")
        self.assertEqual(count, 42)

    async def test_get_document_by_id(self):
        sample_doc = {"_id": ObjectId(), "field": "value"}
        self.repo.serialize_document = lambda doc: doc
        self.repo.db["test"].find_one = AsyncMock(return_value=sample_doc)

        result = await self.repo.get_document_by_id("test", str(sample_doc["_id"]))
        self.assertEqual(result, sample_doc)

    async def test_text_search(self):
        sample_docs = [{"text": "foo"}, {"text": "bar"}]
        cursor_mock = MagicMock()
        cursor_mock.to_list = AsyncMock(return_value=sample_docs)
        self.repo.db["test"].find.return_value.limit.return_value = cursor_mock

        self.repo.serialize_document = lambda doc: doc
        result = await self.repo.text_search("test", "foo")
        self.assertEqual(result, sample_docs)
