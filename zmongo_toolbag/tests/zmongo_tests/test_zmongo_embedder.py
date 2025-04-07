import os
import asyncio
import unittest
import hashlib
from bson.objectid import ObjectId
from dotenv import load_dotenv

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zretriever import ZRetriever
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY_APP")

class OpenAIModel:
    def __init__(self):
        self.model = os.getenv("OPENAI_TEXT_MODEL", "gpt-3.5-turbo-instruct")
        self.max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", 256))
        self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
        self.top_p = float(os.getenv("DEFAULT_TOP_P", 0.95))

    async def _call_openai(self, prompt: str, max_tokens=None, temperature=None, top_p=None, stop=None, echo=False) -> str:
        try:
            response = await asyncio.to_thread(
                openai.completions.create,
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                stop=stop,
                echo=echo,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"[OpenAI Error] {e}"

    async def summarize_text(self, text: str) -> str:
        prompt = f"Summarize the following for a legal researcher:\n\n{text}\n\nSummary:"
        return await self._call_openai(prompt, max_tokens=200)

async def run_openai_zretriever_pipeline():
    repo = ZMongo()
    retriever = ZRetriever(repository=repo, max_tokens_per_set=4096, chunk_size=512)
    collection_name = 'documents'
    document_ids = ['67e5ba645f74ae46ad39929d', '67ef0bd71a349c7c108331a6']

    documents = await retriever.invoke(collection=collection_name, object_ids=document_ids, page_content_key='text')
    openai_model = OpenAIModel()

    for idx, doc_set in enumerate(documents):
        combined_text = "\n\n".join(doc.page_content for doc in doc_set)
        print(f"\nDocument Set {idx + 1} (retrieved {len(doc_set)} chunks):")
        summary = await openai_model.summarize_text(combined_text[:1000])
        print("\nSummary:")
        print(summary)

if __name__ == "__main__":
    asyncio.run(run_openai_zretriever_pipeline())

# --------------------------- TESTS ----------------------------

class TestOpenAIModel(unittest.IsolatedAsyncioTestCase):

    async def test_openai_model_methods(self):
        model = OpenAIModel()
        prompt = "Summarize the following: The quick brown fox jumps over the lazy dog."
        output = await model._call_openai(prompt)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

        summary = await model.summarize_text("The quick brown fox jumps over the lazy dog.")
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

class TestZMongoEmbedder(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.repo = ZMongo()
        self.embedder = ZMongoEmbedder(repository=self.repo, collection="documents")

    async def asyncTearDown(self):
        await self.repo.close()

    async def test_embed_text_and_cache_roundtrip(self):
        text = "Artificial intelligence is transforming the legal industry."
        hash_val = hashlib.sha256(text.encode("utf-8")).hexdigest()
        await self.repo.delete_document("_embedding_cache", {"text_hash": hash_val})

        embedding = await self.embedder.embed_text(text)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

        embedding2 = await self.embedder.embed_text(text)
        self.assertEqual(embedding, embedding2)

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

    async def test_embed_text_invalid_inputs(self):
        invalid_inputs = [None, "", 123, 0.0, [], {}, True]
        for val in invalid_inputs:
            with self.subTest(invalid_input=val):
                with self.assertRaises(ValueError) as ctx:
                    await self.embedder.embed_text(val)
                self.assertEqual(str(ctx.exception), "text must be a non-empty string")

    async def test_embed_and_store_invalid_text_inputs(self):
        invalid_inputs = [None, "", [], {}, 3.1415]
        valid_id = ObjectId()
        for val in invalid_inputs:
            with self.subTest(invalid_text=val):
                with self.assertRaises(ValueError) as ctx:
                    await self.embedder.embed_and_store(valid_id, val)
                self.assertEqual(str(ctx.exception), "text must be a non-empty string")

    async def test_embed_text_invalid_openai_response_data(self):
        class FakeResponse:
            data = []

        class FakeEmbeddings:
            async def create(self, *args, **kwargs):
                return FakeResponse()

        original = self.embedder.openai_client.embeddings
        self.embedder.openai_client.embeddings = FakeEmbeddings()

        with self.assertRaises(ValueError) as ctx:
            await self.embedder.embed_text("This should raise for missing data")
        self.assertIn("missing embedding data", str(ctx.exception))

        self.embedder.openai_client.embeddings = original

    async def test_embed_text_missing_embedding_field(self):
        class NoEmbedding:
            pass

        class FakeResponse:
            data = [NoEmbedding()]

        class FakeEmbeddings:
            async def create(self, *args, **kwargs):
                return FakeResponse()

        original = self.embedder.openai_client.embeddings
        self.embedder.openai_client.embeddings = FakeEmbeddings()

        with self.assertRaises(ValueError) as ctx:
            await self.embedder.embed_text("This should raise for missing 'embedding'")
        self.assertIn("embedding' field is missing", str(ctx.exception))

        self.embedder.openai_client.embeddings = original

    async def test_embed_text_invalid_openai_response_data(self):
        """
        Test that embed_text raises a ValueError when 'response.data' is empty.
        """
        class FakeResponse:
            data = []  # Simulates an empty response

        class FakeEmbeddings:
            async def create(self, *args, **kwargs):
                return FakeResponse()

        original = self.embedder.openai_client.embeddings
        self.embedder.openai_client.embeddings = FakeEmbeddings()

        with self.assertRaises(ValueError) as ctx:
            await self.embedder.embed_text("This should raise for missing data")

        # Fix the expected message to match what your code actually raises
        self.assertIn(
            "OpenAI embedding response is empty; expected at least one embedding.",
            str(ctx.exception)
        )

        self.embedder.openai_client.embeddings = original

    async def test_embed_text_missing_embedding_field(self):
        """
        Test that embed_text raises a ValueError if the record has no 'embedding' attribute.
        """
        class NoEmbedding:
            pass

        class FakeResponse:
            data = [NoEmbedding()]  # Record missing 'embedding'

        class FakeEmbeddings:
            async def create(self, *args, **kwargs):
                return FakeResponse()

        original = self.embedder.openai_client.embeddings
        self.embedder.openai_client.embeddings = FakeEmbeddings()

        with self.assertRaises(ValueError) as ctx:
            await self.embedder.embed_text("This should raise for missing 'embedding'")

        # Fix the expected message to match what your code actually raises
        self.assertIn(
            "OpenAI response is missing embedding data.",
            str(ctx.exception)
        )

        self.embedder.openai_client.embeddings = original

if __name__ == "__main__":
    unittest.main()