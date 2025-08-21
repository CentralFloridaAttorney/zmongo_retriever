import asyncio
import unittest
from bson.objectid import ObjectId
from pymongo.operations import InsertOne, UpdateOne, DeleteOne

from zmongo_retriever.zmongo_toolbag import ZMongo


# Assuming zmongo.py is in the same directory or accessible in the path


class TestZMongoIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration test suite for the ZMongo class.
    This suite runs against a REAL MongoDB instance defined in your .env_local file.
    """

    async def asyncSetUp(self):
        """
        Set up a new ZMongo instance and a unique collection for each test.
        """
        self.zm = ZMongo()
        self.db = self.zm.db
        self.collection_name = self._testMethodName
        if self.db is not None:
            await self.db.drop_collection(self.collection_name)

    async def asyncTearDown(self):
        """
        Clean up by dropping the test collection and closing the connection.
        """
        if self.db is not None and self.collection_name:
            await self.db.drop_collection(self.collection_name)
        self.zm.close()

    async def test_insert_and_find_document(self):
        """Test inserting a single document and then finding it."""
        doc = {"name": "test_doc", "value": 123}
        insert_res = await self.zm.insert_document(self.collection_name, doc)
        self.assertTrue(insert_res.success)
        self.assertIsNotNone(insert_res.data['inserted_id'])
        doc_id = ObjectId(insert_res.data['inserted_id'])

        find_res = await self.zm.find_document(self.collection_name, {"_id": doc_id})
        self.assertTrue(find_res.success)
        self.assertEqual(find_res.data["name"], "test_doc")
        self.assertEqual(str(doc_id), find_res.data["_id"])

    async def test_insert_many_and_count(self):
        """Test inserting multiple documents and verifying the count."""
        docs = [{"name": f"doc_{i}"} for i in range(10)]
        insert_res = await self.zm.insert_documents(self.collection_name, docs)
        self.assertTrue(insert_res.success)
        self.assertEqual(len(insert_res.data['inserted_ids']), 10)

        count_res = await self.zm.count_documents(self.collection_name, {})
        self.assertTrue(count_res.success)
        self.assertEqual(count_res.data["count"], 10)

    async def test_caching_behavior(self):
        """Test that the cache serves stale data until invalidated."""
        doc = {"name": "cached_doc", "version": 1}
        insert_res = await self.zm.insert_document(self.collection_name, doc)
        doc_id = ObjectId(insert_res.data['inserted_id'])

        find_res_1 = await self.zm.find_document(self.collection_name, {"_id": doc_id}, cache=True)
        self.assertEqual(find_res_1.data["version"], 1)

        await self.db[self.collection_name].update_one({"_id": doc_id}, {"$set": {"version": 2}})

        find_res_2_cached = await self.zm.find_document(self.collection_name, {"_id": doc_id}, cache=True)
        self.assertEqual(find_res_2_cached.data["version"], 1, "Should serve stale data from cache")

        find_res_3_fresh = await self.zm.find_document(self.collection_name, {"_id": doc_id}, cache=False)
        self.assertEqual(find_res_3_fresh.data["version"], 2, "Should fetch fresh data from DB")

    async def test_cache_invalidation_on_update(self):
        """Test that zm.update_document correctly invalidates the cache."""
        doc = {"name": "doc_to_update", "version": 1}
        insert_res = await self.zm.insert_document(self.collection_name, doc)
        doc_id = ObjectId(insert_res.data['inserted_id'])

        await self.zm.find_document(self.collection_name, {"_id": doc_id})
        await self.zm.update_document(self.collection_name, {"_id": doc_id}, {"version": 2})

        find_res = await self.zm.find_document(self.collection_name, {"_id": doc_id})
        self.assertEqual(find_res.data["version"], 2)

    async def test_update_and_delete_document(self):
        """Test updating and then deleting a document."""
        insert_res = await self.zm.insert_document(self.collection_name, {"name": "test", "status": "active"})
        doc_id = ObjectId(insert_res.data['inserted_id'])

        update_res = await self.zm.update_document(self.collection_name, {"_id": doc_id}, {"status": "inactive"})
        self.assertTrue(update_res.success)
        # --- FIX: Use dictionary access ---
        self.assertEqual(update_res.data['modified_count'], 1)

        updated_doc = await self.db[self.collection_name].find_one({"_id": doc_id})
        self.assertEqual(updated_doc["status"], "inactive")

        delete_res = await self.zm.delete_document(self.collection_name, {"_id": doc_id})
        self.assertTrue(delete_res.success)
        # --- FIX: Use dictionary access ---
        self.assertEqual(delete_res.data['deleted_count'], 1)

        deleted_doc = await self.db[self.collection_name].find_one({"_id": doc_id})
        self.assertIsNone(deleted_doc)

    async def test_bulk_write_operation(self):
        """Test a mixed bulk write operation."""
        await self.db[self.collection_name].insert_many([
            {"_id": ObjectId("607f191e810c19729de860ea"), "name": "doc_one"},
            {"_id": ObjectId("607f1f77bcf86cd799439011"), "name": "doc_two"},
        ])

        ops = [
            InsertOne({"name": "doc_three"}),
            UpdateOne({"name": "doc_one"}, {"$set": {"value": 10}}),
            DeleteOne({"name": "doc_two"})
        ]

        bulk_res = await self.zm.bulk_write(self.collection_name, ops)
        self.assertTrue(bulk_res.success)
        # --- FIX: Use dictionary access for all assertions ---
        self.assertEqual(bulk_res.data['inserted_count'], 1)
        self.assertEqual(bulk_res.data['modified_count'], 1)
        self.assertEqual(bulk_res.data['deleted_count'], 1)

        count = await self.db[self.collection_name].count_documents({})
        self.assertEqual(count, 2)
        doc_one = await self.db[self.collection_name].find_one({"name": "doc_one"})
        self.assertEqual(doc_one["value"], 10)

    async def test_aggregation_pipeline(self):
        """Test a simple aggregation pipeline."""
        docs = [
            {"category": "A", "value": 10},
            {"category": "B", "value": 20},
            {"category": "A", "value": 15},
        ]
        await self.zm.insert_documents(self.collection_name, docs)

        pipeline = [
            {"$group": {"_id": "$category", "total": {"$sum": "$value"}}},
            {"$sort": {"_id": 1}}
        ]

        agg_res = await self.zm.aggregate(self.collection_name, pipeline)
        self.assertTrue(agg_res.success)
        self.assertEqual(len(agg_res.data), 2)
        self.assertEqual(agg_res.data[0], {"_id": "A", "total": 25})
        self.assertEqual(agg_res.data[1], {"_id": "B", "total": 20})

    async def test_list_collections(self):
        """Test listing collections in the database."""
        await self.zm.insert_document(f"{self.collection_name}_1", {"a": 1})
        await self.zm.insert_document(f"{self.collection_name}_2", {"b": 1})

        list_res = await self.zm.list_collections()
        self.assertTrue(list_res.success)
        self.assertIn(f"{self.collection_name}_1", list_res.data)
        self.assertIn(f"{self.collection_name}_2", list_res.data)

        await self.db.drop_collection(f"{self.collection_name}_1")
        await self.db.drop_collection(f"{self.collection_name}_2")

if __name__ == '__main__':
    unittest.main()