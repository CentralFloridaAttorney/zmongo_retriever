"""
demo_retriever_from_tests.py
----------------------------
A runnable demonstration that mirrors the integration flows defined in
tests/zmongo_tests/test_zmongo_retriever_facts.py, but as a single script.

Requirements:
  - MongoDB reachable via MONGO_URI (or defaults to localhost)
  - GEMINI_API_KEY set (env or ~/resources/.env_local)
Usage:
  python demo_retriever_from_tests.py
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List

from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document

from zmongo import ZMongo
from zmongo_embedder import ZMongoEmbedder
from unified_vector_search import LocalVectorSearch
from zmongo_retriever import ZMongoRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("demo_from_tests")

# Match the test’s env-file convention
load_dotenv(Path.home() / "resources" / ".env_local")

COLLECTION_NAME = "retriever_test_coll"


class DemoZMongoRetrieverFacts:
    """
    A demonstration harness that reproduces the core behaviors from
    test_zmongo_retriever_facts.py as interactive examples.
    """

    def __init__(self):
        # Basic env checks (same assumptions as tests)
        if not os.getenv("GEMINI_API_KEY"):
            raise SystemExit("ERROR: GEMINI_API_KEY is not set.")
        # MONGO_URI is optional—ZMongo falls back to localhost if unset.

        self.repo = ZMongo()
        self.embedder = ZMongoEmbedder(collection=COLLECTION_NAME)
        self.vector_searcher = LocalVectorSearch(
            repository=self.repo,
            collection=COLLECTION_NAME,
            embedding_field="embeddings",
            chunked_embeddings=True,
            exact_rescore=True,
        )
        # Mirror test defaults where sensible
        self.retriever = ZMongoRetriever(
            repository=self.repo,
            embedder=self.embedder,
            vector_searcher=self.vector_searcher,
            collection_name=COLLECTION_NAME,
            similarity_threshold=0.0,
            top_k=5,
        )

    async def _populate(self, documents: List[dict]):
        """
        Equivalent to the test helper populate_test_data(): chunk-embeds texts first,
        attaches embeddings to docs, then inserts in bulk.
        """
        texts = [d.get("text") for d in documents if d.get("text")]
        if texts:
            embs_map = await self.embedder.embed_texts_batched(texts)
            for d in documents:
                text = d.get("text")
                if text and text in embs_map:
                    d["embeddings"] = embs_map[text]
        await self.repo.insert_documents(COLLECTION_NAME, documents)
        # Small pause to ensure cluster visibility when running against remote Mongo
        await asyncio.sleep(0.5)

    async def reset_collection(self):
        await self.repo.delete_documents(COLLECTION_NAME, {})

    # ---------- Demo segments (mirroring the tests) ----------

    async def demo_initialization(self):
        print("\n[demo_initialization]")
        print("Repository:", type(self.retriever.repository).__name__)
        print("Embedder:", type(self.retriever.embedder).__name__)
        print("Vector Searcher:", type(self.retriever.vector_searcher).__name__)
        print("Collection:", self.retriever.collection_name)

    async def demo_retrieval_flow_with_filtering(self):
        print("\n[demo_retrieval_flow_with_filtering]")
        await self.reset_collection()
        docs = [
            {"_id": ObjectId(), "text": "Python is a versatile programming language."},
            {"_id": ObjectId(), "text": "The sky is blue and the grass is green."},
            {"_id": ObjectId(), "text": "A dynamic, high-level, object-oriented language is Python."},
        ]
        await self._populate(docs)

        query = "What is a good programming language?"
        results = await self.retriever.ainvoke(query)
        self._print_results(results)

    async def demo_document_formatting_and_metadata(self):
        print("\n[demo_document_formatting_and_metadata]")
        await self.reset_collection()
        doc_id = ObjectId()
        test_doc = {
            "_id": doc_id,
            "text": "This is the main content.",
            "author": "Test Author",
            "category": "Testing",
        }
        await self._populate([test_doc])

        results = await self.retriever.ainvoke("A query for the main content")
        self._print_results(results, show_full_metadata=True)

    async def demo_no_results_found(self):
        print("\n[demo_no_results_found]")
        await self.reset_collection()
        results = await self.retriever.ainvoke("Query with no possible results")
        self._print_results(results)

    async def demo_retrieval_with_distinct_facts(self):
        print("\n[demo_retrieval_with_distinct_facts]")
        await self.reset_collection()
        kb = [
            {
                "_id": ObjectId(),
                "topic": "Astronomy",
                "text": (
                    "Jupiter is the fifth planet from the Sun and the largest in the Solar System. "
                    "It is a gas giant with a mass more than two and a half times that of all the "
                    "other planets combined."
                ),
            },
            {
                "_id": ObjectId(),
                "topic": "Biology",
                "text": (
                    "Mitochondria are organelles that act like a digestive system which takes in "
                    "nutrients, breaks them down, and creates energy rich molecules for the cell. "
                    "They are often referred to as the powerhouse of the cell."
                ),
            },
            {
                "_id": ObjectId(),
                "topic": "History",
                "text": (
                    "The Roman Empire was one of the most influential civilizations in world history, "
                    "known for its contributions to law, architecture, and language, lasting for over "
                    "a thousand years."
                ),
            },
        ]
        await self._populate(kb)

        query = "What is the powerhouse of the cell?"
        results = await self.retriever.ainvoke(query)
        self._print_results(results)

    # ---------- Utility printing ----------

    def _print_results(self, results: List[Document], show_full_metadata: bool = False):
        if not results:
            print("No documents found (or none met the similarity threshold).")
            return

        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}")
            print("-" * 40)
            print("Content:")
            print(doc.page_content)
            print("\nMetadata:")
            if show_full_metadata:
                for k, v in doc.metadata.items():
                    print(f"  {k}: {v}")
            else:
                # Compact view: highlight score & id; hide large fields if any
                rid = doc.metadata.get("_id")
                score = doc.metadata.get("retrieval_score")
                topic = doc.metadata.get("topic")
                author = doc.metadata.get("author")
                print(f"  _id: {rid}")
                if topic:
                    print(f"  topic: {topic}")
                if author:
                    print(f"  author: {author}")
                print(f"  retrieval_score: {score}")

    # ---------- Run all demos ----------

    async def run_all(self):
        await self.demo_initialization()
        await self.demo_retrieval_flow_with_filtering()
        await self.demo_document_formatting_and_metadata()
        await self.demo_no_results_found()
        await self.demo_retrieval_with_distinct_facts()


async def main():
    demo = DemoZMongoRetrieverFacts()
    await demo.run_all()


if __name__ == "__main__":
    asyncio.run(main())
