import unittest
import asyncio
from bson import ObjectId

from zmongo_toolbag.safe_result import SafeResult
from zmongo_toolbag.zmongo import ZMongo

class TestZMongoExceptionHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.collection = "test_exception"
        await self.repo.delete_all_documents(self.collection)

    async def asyncTearDown(self):
        await self.repo.delete_all_documents(self.collection)
        await self.repo.close()

    async def test_insert_document_invalid(self):
        # Insert document with unserializable field (like a function)
        doc = {"foo": lambda x: x}
        res = await self.repo.insert_document(self.collection, doc)
        data = res.model_dump()
        self.assertIn("error", data)
        self.assertIsNone(data.get("inserted_id"))

    async def test_insert_documents_invalid(self):
        # Use a batch with an unserializable item
        docs = [{"bar": object()}, {"baz": "valid"}]
        res = await self.repo.insert_documents(self.collection, docs)
        data = res.model_dump()
        self.assertIn("errors", data)
        self.assertIsInstance(data["inserted_ids"], list)

    async def test_update_document_exception(self):
        # Use invalid update data (like a set)
        doc = {"_id": ObjectId(), "v": 1}
        await self.repo.insert_document(self.collection, doc)
        res = await self.repo.update_document(self.collection, {"_id": doc["_id"]}, {"new": set([1,2,3])})
        data = res.model_dump()
        self.assertIn("error", data)
        self.assertEqual(data["matched_count"], 0)

    async def test_delete_documents_invalid_collection(self):
        # Use a likely "weird" name, but don't expect Mongo to error unless truly illegal
        res = await self.repo.delete_documents("!!!illegal_collection", {})
        data = res.model_dump()
        self.assertIsInstance(res, SafeResult)
        self.assertIn("deleted_count", data)
        self.assertEqual(data["deleted_count"], 0)
        # If you ever expect error handling, check like this:
        # if "error" in data:
        #     self.assertIsInstance(data["error"], str)


    async def test_delete_document_invalid_query(self):
        # Use a query with an unserializable field
        res = await self.repo.delete_document(self.collection, {"bad": object()})
        data = res.model_dump()
        self.assertIn("error", data)
        self.assertEqual(data["deleted_count"], 0)

    async def test_bulk_write_errors(self):
        # Use invalid operation (like a dict instead of UpdateOne/InsertOne)
        ops = [{"illegal": "op"}]
        res = await self.repo.bulk_write(self.collection, ops)
        data = res.model_dump()
        self.assertIn("error", data)

    async def test_text_search_no_index(self):
        # Trigger $text search on a collection without a text index
        await self.repo.insert_document(self.collection, {"content": "foo"})
        res = await self.repo.text_search(self.collection, "foo")
        self.assertIsInstance(res, SafeResult)
        data = res.model_dump()
        self.assertIsInstance(data, list)
        # Optionally, you can check it's empty (if you know it should be)
        # self.assertEqual(data, [])

    async def test_count_documents_invalid(self):
        # Use a collection name that does not exist
        res = await self.repo.count_documents("bad!@#$name")
        self.assertIn("count", res.model_dump())

    async def test_save_embedding_invalid(self):
        # Use a non-serializable embedding
        doc_id = ObjectId()
        emb = [object()]
        res = await self.repo.save_embedding(self.collection, doc_id, emb)
        data = res.model_dump()
        self.assertIn("error", data)
        self.assertFalse(data["saved"])

    async def test_get_document_by_id_invalid(self):
        # Use an invalid ObjectId string
        res = await self.repo.get_document_by_id(self.collection, "not_an_objectid")
        self.assertIsNone(res.model_dump())

    async def test_get_field_names_invalid(self):
        # Use a collection that doesn't exist
        res = await self.repo.get_field_names("nope_nope_nope")
        self.assertIsInstance(res.model_dump(), list)

    async def test_log_training_metrics_error(self):
        # Use a metrics dict with unserializable value
        res = self.repo.log_training_metrics({"fail": object()})
        data = res.model_dump()
        self.assertIn("error", data)
        self.assertFalse(data["logged"])

if __name__ == "__main__":
    asyncio.run(unittest.main())
