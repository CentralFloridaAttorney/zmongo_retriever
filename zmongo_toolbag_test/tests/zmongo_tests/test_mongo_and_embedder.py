import unittest
import asyncio

from bson.objectid import ObjectId


class TestZMongoAndEmbedder(unittest.TestCase):

    async def test_embed_text(self):
        text = "some text"

        # Ensure that calling the embed_text method calls the find_document method properly
        result = await self.repo.embed_text(text)

        # Perform assertions as needed
        self.assertEqual(result, "expected_result")  # Example assertion, modify as per your logic

    # Other tests here...

    def tearDown(self):
        # This may no longer be needed if unittest automatically manages event loops.
        # However, if required to manage the event loop manually, consider using `asyncio` to close it.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
            loop.close()
        except RuntimeError:
            # Handle the case where no loop is found
            pass


    async def test_embed_and_store(self):
        text = "AI in courtrooms can help with evidence organization."
        document = {"text": text, "label": "test_embed"}

        # Insert document and get the InsertOneResult
        result = await self.repo.insert_document("documents", document)

        # FIX: extract _id from the InsertOneResult
        _id = result.inserted_id  # Access the inserted_id from InsertOneResult
        self.assertIsInstance(_id, ObjectId)  # Safety check

        # Now embed and store
        await self.embedder.embed_and_store(_id, text)

        # Verify it was saved
        updated_doc = await self.repo.find_document("documents", {"_id": _id})
        self.assertIn("embedding", updated_doc)
        self.assertIsInstance(updated_doc["embedding"], list)
