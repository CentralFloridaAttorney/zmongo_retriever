import asyncio
import os
import unittest
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo import ZMongo  # Adjust import path to your actual project structure

class TestZMongoCacheIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ["MONGO_URI"] = "mongodb://localhost:27017"
        os.environ["MONGO_DATABASE_NAME"] = "test_zmongo_cache"
        self.zmongo = ZMongo()
        self.collection = "test_collection"
        await self.zmongo.delete_all_documents(self.collection)
        await self.zmongo.clear_cache()

    async def asyncTearDown(self):
        await self.zmongo.delete_all_documents(self.collection)
        await self.zmongo.clear_cache()
        await self.zmongo.close()

    async def test_update_document_updates_cache(self):
        # Insert a document
        document = {"name": "John", "age": 30}
        insert_result = await self.zmongo.insert_document(self.collection, document)
        self.assertIsNotNone(insert_result)
        inserted_id = insert_result.inserted_id

        # Confirm document is inserted
        query = {"_id": inserted_id}
        found = await self.zmongo.find_document(self.collection, query)
        self.assertIsNotNone(found)
        self.assertEqual(found["name"], "John")
        self.assertEqual(found["age"], 30)

        # Confirm it is now cached
        normalized = self.zmongo._normalize_collection_name(self.collection)
        cache_key = self.zmongo._generate_cache_key({"_id": str(inserted_id)})
        self.assertIn(cache_key, self.zmongo.cache[normalized])
        self.assertEqual(self.zmongo.cache[normalized][cache_key]["age"], 30)

        # Update the document
        update = {"$set": {"age": 31}}
        result = await self.zmongo.update_document(self.collection, query, update)
        self.assertEqual(result.modified_count, 1)

        # Ensure the cache reflects the updated value
        updated_doc = self.zmongo.cache[normalized][cache_key]
        self.assertEqual(updated_doc["age"], 31)

        # Also confirm via a fresh query
        fetched_again = await self.zmongo.find_document(self.collection, query)
        self.assertEqual(fetched_again["age"], 31)


if __name__ == "__main__":
    unittest.main()
