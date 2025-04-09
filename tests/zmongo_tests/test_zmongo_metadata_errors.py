import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoMetadataErrors(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = ZMongo()
        self.repo.db = MagicMock()
        self.repo.serialize_document = lambda doc: doc  # No-op for simplicity

    async def test_get_field_names_error(self):
        self.repo.db["bad_collection"].find.side_effect = Exception("DB error")
        result = await self.repo.get_field_names("bad_collection")
        self.assertEqual(result, [])  # Should return empty list on failure

    async def test_sample_documents_error(self):
        self.repo.db["sample"].find.side_effect = Exception("Read failure")
        result = await self.repo.sample_documents("sample")
        self.assertEqual(result, [])  # Should return empty list on failure

    async def test_count_documents_error(self):
        self.repo.db["docs"].estimated_document_count = AsyncMock(side_effect=Exception("Count fail"))
        result = await self.repo.count_documents("docs")
        self.assertEqual(result, 0)  # Should return 0 on failure

    async def test_get_document_by_id_invalid_id(self):
        # Pass a bad ObjectId string
        result = await self.repo.get_document_by_id("test", "not_an_objectid")
        self.assertIsNone(result)

    async def test_get_document_by_id_lookup_failure(self):
        oid = ObjectId()
        self.repo.db["test"].find_one = AsyncMock(side_effect=Exception("Lookup failure"))
        result = await self.repo.get_document_by_id("test", oid)
        self.assertIsNone(result)

    async def test_text_search_error(self):
        self.repo.db["text"].find.side_effect = Exception("Text search issue")
        result = await self.repo.text_search("text", "anything")
        self.assertEqual(result, [])  # Should return empty list

if __name__ == "__main__":
    unittest.main()
