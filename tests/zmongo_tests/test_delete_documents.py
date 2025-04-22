import asyncio
import unittest
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo

class TestZMongoDeleteDocument(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.collection = "test_delete_single"
        # Clean collection
        await self.repo.delete_all_documents(self.collection)

    async def asyncTearDown(self):
        await self.repo.delete_all_documents(self.collection)
        await self.repo.close()

    async def test_delete_document_deletes_only_one(self):
        # Insert three documents
        docs = [
            {"_id": ObjectId(), "val": 1},
            {"_id": ObjectId(), "val": 2},
            {"_id": ObjectId(), "val": 3},
        ]
        await self.repo.insert_documents(self.collection, docs)
        # Delete the document with val == 2
        target_id = docs[1]["_id"]
        res = await self.repo.delete_document(self.collection, {"_id": target_id})
        self.assertEqual(res.model_dump()["deleted_count"], 1)
        # Check that only the target doc is gone
        remaining = await self.repo.find_documents(self.collection, {})
        remaining_ids = [doc["_id"] for doc in remaining.model_dump()]
        self.assertNotIn(str(target_id), remaining_ids)
        self.assertIn(str(docs[0]["_id"]), remaining_ids)
        self.assertIn(str(docs[2]["_id"]), remaining_ids)
        self.assertEqual(len(remaining_ids), 2)

if __name__ == "__main__":
    asyncio.run(unittest.main())
