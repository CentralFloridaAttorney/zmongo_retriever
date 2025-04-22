import unittest
import asyncio
import os
import random
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo, SafeResult

class TestZMongoFullCoverage(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Use a test DB name to avoid polluting production DB
        os.environ["MONGO_DATABASE_NAME"] = "test_zmongo_full_coverage"
        cls.collection = "test_zmongo_collection"

    async def asyncSetUp(self):
        self.repo = ZMongo()
        await self.repo.delete_all_documents(self.collection)
        self.other_collection = "other_collection"
        await self.repo.delete_all_documents(self.other_collection)

    async def asyncTearDown(self):
        await self.repo.delete_all_documents(self.collection)
        await self.repo.delete_all_documents(self.other_collection)
        await self.repo.clear_cache()
        await self.repo.close()

    async def test_insert_and_find_document(self):
        doc = {"_id": ObjectId(), "name": "foo"}
        # insert_document
        result = await self.repo.insert_document(self.collection, doc)
        self.assertIsInstance(result, SafeResult)
        out = result.model_dump()
        self.assertEqual(out["inserted_id"], str(doc["_id"]))

        # find_document (cache miss)
        found = await self.repo.find_document(self.collection, {"_id": doc["_id"]})
        self.assertIsInstance(found, SafeResult)
        found_doc = found.model_dump()
        self.assertEqual(found_doc["_id"], doc["_id"])

        # find_document (cache hit)
        found2 = await self.repo.find_document(self.collection, {"_id": doc["_id"]})
        self.assertEqual(found2.model_dump()["_id"], doc["_id"])

    async def test_insert_documents_and_bulk_write(self):
        docs = [{"_id": ObjectId(), "x": i} for i in range(5)]
        result = await self.repo.insert_documents(self.collection, docs)
        out = result.model_dump()
        self.assertEqual(len(out["inserted_ids"]), 5)

        # bulk_write success
        from pymongo import InsertOne, DeleteOne
        ops = [InsertOne({"_id": ObjectId(), "a": 1}), DeleteOne({"_id": docs[0]["_id"]})]
        bw_result = await self.repo.bulk_write(self.collection, ops)
        self.assertIn("acknowledged", bw_result.model_dump())

        # bulk_write with empty ops
        empty_result = await self.repo.bulk_write(self.collection, [])
        self.assertEqual(empty_result.model_dump()["inserted_count"], 0)

    async def test_update_document(self):
        doc = {"_id": ObjectId(), "value": 10}
        await self.repo.insert_document(self.collection, doc)
        upd_result = await self.repo.update_document(
            self.collection, {"_id": doc["_id"]}, {"value": 20}
        )
        out = upd_result.model_dump()
        self.assertTrue(out["matched_count"] >= 1)
        # Confirm update
        found = await self.repo.find_document(self.collection, {"_id": doc["_id"]})
        self.assertEqual(found.model_dump()["value"], 20)

    async def test_delete_documents_and_delete_all(self):
        docs = [{"_id": ObjectId(), "z": i} for i in range(3)]
        await self.repo.insert_documents(self.collection, docs)
        # delete_documents
        ids = [d["_id"] for d in docs[:2]]
        del_res = await self.repo.delete_documents(self.collection, {"_id": {"$in": ids}})
        self.assertIn("deleted_count", del_res)
        # delete_all_documents
        del_all = await self.repo.delete_all_documents(self.collection)
        self.assertIn("deleted_count", del_all.model_dump())

    async def test_save_embedding_and_get_document_by_id(self):
        doc = {"_id": ObjectId(), "text": "abc"}
        await self.repo.insert_document(self.collection, doc)
        # save_embedding
        emb = [random.random() for _ in range(5)]
        se_res = await self.repo.save_embedding(self.collection, doc["_id"], emb)
        self.assertTrue(se_res.model_dump()["saved"])
        # get_document_by_id
        found = await self.repo.get_document_by_id(self.collection, str(doc["_id"]))
        # self.assertEqual(found.model_dump()["_id"], doc["_id"])
        expected_id = str(doc["_id"])
        actual_id = found.model_dump()["_id"]
        self.assertEqual(actual_id, expected_id)

        # get_document_by_id with invalid id
        inv = await self.repo.get_document_by_id(self.collection, "not_a_real_objectid")
        self.assertIsNone(inv.model_dump())

    async def test_clear_cache_and_list_collections(self):
        await self.repo.clear_cache()
        res = await self.repo.list_collections()
        self.assertIsInstance(res, SafeResult)

    async def test_sample_documents_and_text_search(self):
        docs = [{"_id": ObjectId(), "txt": f"word{i}"} for i in range(10)]
        await self.repo.insert_documents(self.collection, docs)
        # sample_documents
        s_res = await self.repo.sample_documents(self.collection, sample_size=3)
        self.assertTrue(len(s_res.model_dump()) <= 3)
        # text_search with no text index (should not error)
        ts = await self.repo.text_search(self.collection, "foo")
        self.assertIsInstance(ts, SafeResult)

    async def test_count_documents_and_log_training_metrics(self):
        docs = [{"_id": ObjectId(), "val": i} for i in range(5)]
        await self.repo.insert_documents(self.collection, docs)
        count = await self.repo.count_documents(self.collection)
        self.assertTrue(count.model_dump()["count"] >= 5)
        # log_training_metrics (sync)
        result = self.repo.log_training_metrics({"foo": 1})
        self.assertTrue(result.model_dump()["logged"])

    async def test_insert_documents_sync(self):
        docs = [{"y": 1}, {"y": 2}]
        out = self.repo.insert_documents_sync(self.collection, docs)
        self.assertIn("inserted_ids", out)

    async def test_error_branches(self):
        # Find non-existent
        not_found = await self.repo.find_document(self.collection, {"_id": ObjectId()})
        self.assertIsNone(not_found.model_dump())
        # Text search error (simulate by passing weird input)
        ts = await self.repo.text_search(self.collection, "\x00\x00\x00")
        self.assertIsInstance(ts, SafeResult)
        # Count error on non-existent collection
        _ = await self.repo.count_documents("collection_does_not_exist_for_sure")

if __name__ == "__main__":
    unittest.main()
