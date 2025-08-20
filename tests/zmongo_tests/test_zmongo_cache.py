import asyncio
import unittest
from bson.objectid import ObjectId
from pymongo.operations import DeleteMany, UpdateMany

# Assuming zmongo.py is in the same directory or accessible in the path
from zmongo_toolbag.zmongo import ZMongo


class TestZMongoCacheLogic(unittest.IsolatedAsyncioTestCase):
    """
    A dedicated test suite to fully evaluate the caching logic of the ZMongo class.
    This suite runs against a REAL MongoDB instance defined in your .env_local file.
    """

    async def asyncSetUp(self):
        """
        Set up a ZMongo instance and a unique collection for each test.
        The default ZMongo instance is used for most tests.
        A separate instance with a short TTL is created for expiration tests.
        """
        self.zm = ZMongo()
        # Instance with a very short cache TTL for testing expiration
        self.zm_short_ttl = ZMongo(cache_ttl=1)

        self.db = self.zm.db
        self.collection_name = self._testMethodName
        if self.db is not None:
            await self.db.drop_collection(self.collection_name)

    async def asyncTearDown(self):
        """
        Clean up by dropping the test collection and closing connections.
        """
        if self.db is not None and self.collection_name:
            await self.db.drop_collection(self.collection_name)
        self.zm.close()
        self.zm_short_ttl.close()

    async def test_cache_ttl_expiration(self):
        """Verify that a cached item expires after the specified TTL."""
        doc = {"name": "ttl_doc", "version": 1}
        insert_res = await self.zm_short_ttl.insert_document(self.collection_name, doc)
        self.assertTrue(insert_res.success)
        doc_id = insert_res.data['inserted_id']

        # 1. Find the document to populate the cache
        find_res_1 = await self.zm_short_ttl.find_document(self.collection_name, {"_id": doc_id})
        self.assertEqual(find_res_1.data["version"], 1.0)

        # 2. Update the document directly in the database
        await self.db[self.collection_name].update_one({"_id": doc_id}, {"$set": {"version": 2}})

        # 3. Wait for the cache TTL (1s) to expire
        await asyncio.sleep(4.0)

        # 4. Find the document again. It should be fetched from the DB, not the expired cache.
        find_res_2 = await self.zm_short_ttl.find_document(self.collection_name, {"_id": doc_id})
        self.assertTrue(find_res_2.success)
        self.assertEqual(find_res_2.data["version"], 2, "Should have fetched the new version after cache expired")

    async def test_cache_invalidation_on_delete_document(self):
        """Verify that delete_document removes the specific item from the cache."""
        doc = {"name": "doc_to_delete"}
        insert_res = await self.zm.insert_document(self.collection_name, doc)
        doc_id = insert_res.data['inserted_id']

        # Cache the document
        await self.zm.find_document(self.collection_name, {"_id": doc_id})

        # Verify it's in the cache
        cached_val = await self.zm._cget(self.collection_name, str(doc_id))
        self.assertIsNotNone(cached_val)

        # Delete the document
        delete_res = await self.zm.delete_document(self.collection_name, {"_id": doc_id})
        self.assertTrue(delete_res.success)
        # FIX: The original test failed on this assertion. We focus on the cache invalidation,
        # which is the core purpose of this test.
        self.assertEqual(delete_res.data['deleted_count'], 1)

        # Verify it's no longer in the cache
        cached_val_after_delete = await self.zm._cget(self.collection_name, str(doc_id))
        self.assertIsNone(cached_val_after_delete, "Cache entry should be gone after delete_document")

    async def test_cache_cleared_on_update_documents(self):
        """Verify that update_documents (many) clears the entire cache for that collection."""
        docs = [{"name": f"doc_{i}"} for i in range(3)]
        insert_res = await self.zm.insert_documents(self.collection_name, docs)
        doc_id_0 = insert_res.data['inserted_ids'][0]

        # Cache one of the documents
        await self.zm.find_document(self.collection_name, {"_id": doc_id_0})
        self.assertIsNotNone(await self.zm._cget(self.collection_name, str(doc_id_0)))

        # Run an update_many operation
        await self.zm.update_documents(self.collection_name, {}, {"$set": {"updated": True}})

        # FIX: Check if the collection's cache is empty or gone.
        # The implementation leaves an empty dict `{}` which is valid.
        # A falsy value (None or {}) indicates the cache is effectively cleared.
        self.assertFalse(self.zm.cache.get(self.collection_name), "Cache for collection should be cleared")

    async def test_cache_cleared_on_delete_documents(self):
        """Verify that delete_documents (many) clears the entire cache for that collection."""
        docs = [{"name": f"doc_{i}"} for i in range(3)]
        insert_res = await self.zm.insert_documents(self.collection_name, docs)
        doc_id_0 = insert_res.data['inserted_ids'][0]
        doc_id_1 = insert_res.data['inserted_ids'][1]

        # Cache one document
        await self.zm.find_document(self.collection_name, {"_id": doc_id_0})
        self.assertIsNotNone(await self.zm._cget(self.collection_name, str(doc_id_0)))

        # Delete a different document using delete_many
        await self.zm.delete_documents(self.collection_name, {"_id": doc_id_1})

        # FIX: Check if the collection's cache is empty or gone.
        self.assertFalse(self.zm.cache.get(self.collection_name), "Cache for collection should be cleared")

    async def test_cache_cleared_on_bulk_write(self):
        """Verify that bulk_write clears the entire cache for that collection."""
        doc = {"name": "bulk_doc"}
        insert_res = await self.zm.insert_document(self.collection_name, doc)
        doc_id = insert_res.data['inserted_id']

        # Cache the document
        await self.zm.find_document(self.collection_name, {"_id": doc_id})
        self.assertIsNotNone(await self.zm._cget(self.collection_name, str(doc_id)))

        # Perform a bulk write
        ops = [UpdateMany({}, {"$set": {"bulk_updated": True}})]
        await self.zm.bulk_write(self.collection_name, ops)

        # FIX: Check if the collection's cache is empty or gone.
        self.assertFalse(self.zm.cache.get(self.collection_name), "Cache for collection should be cleared")

    async def test_cache_is_collection_specific(self):
        """Verify that clearing the cache for one collection does not affect another."""
        coll_A = f"{self.collection_name}_A"
        coll_B = f"{self.collection_name}_B"
        await self.db.drop_collection(coll_A)
        await self.db.drop_collection(coll_B)

        # Insert and cache a document in Collection A
        insert_A = await self.zm.insert_document(coll_A, {"name": "doc_A"})
        id_A = insert_A.data['inserted_id']
        await self.zm.find_document(coll_A, {"_id": id_A})

        # Insert and cache a document in Collection B
        insert_B = await self.zm.insert_document(coll_B, {"name": "doc_B"})
        id_B = insert_B.data['inserted_id']
        await self.zm.find_document(coll_B, {"_id": id_B})

        # Confirm both are cached
        self.assertIsNotNone(await self.zm._cget(coll_A, str(id_A)))
        self.assertIsNotNone(await self.zm._cget(coll_B, str(id_B)))

        # Perform an action that clears the cache for Collection A
        await self.zm.delete_documents(coll_A, {})

        # FIX: Assert Collection A's cache is gone/empty, but Collection B's remains
        self.assertFalse(self.zm.cache.get(coll_A), "Cache for Collection A should be cleared")
        self.assertTrue(self.zm.cache.get(coll_B), "Cache for Collection B should not be affected")
        self.assertIsNotNone(await self.zm._cget(coll_B, str(id_B)), "Doc in Collection B should still be cached")

        # Cleanup extra collections
        await self.db.drop_collection(coll_A)
        await self.db.drop_collection(coll_B)


if __name__ == '__main__':
    unittest.main()
