"""
demo_langchain_retrieval.py
---------------------------
Minimal end-to-end example showing how to use ZMongoRetriever with LangChain.

Requirements:
  - Python 3.10+
  - A running MongoDB instance (MONGO_URI env var or defaults to localhost)
  - GEMINI_API_KEY set in env or ~/resources/.env_local

Usage:
  python demo_langchain_retrieval.py
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.schema import Document

# Import from your package layout
from zmongo import ZMongo
from zmongo_embedder import ZMongoEmbedder
from unified_vector_search import LocalVectorSearch
from zmongo_retriever import ZMongoRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("demo")

# Load ~/resources/.env_local if present (matches your project convention)
load_dotenv(Path.home() / "resources" / ".env_local")


async def prepare_data(repo: ZMongo, collection: str) -> List[Dict[str, Any]]:
    """
    Clears the collection, inserts a tiny KB, and returns the inserted docs.
    """
    logger.info("Clearing collection: %s", collection)
    await repo.delete_documents(collection, {})

    knowledge_base: List[Dict[str, Any]] = [
        {
            "topic": "Astronomy",
            "text": (
                "Jupiter is the fifth planet from the Sun and the largest in the Solar System. "
                "It is a gas giant."
            ),
        },
        {
            "topic": "Biology",
            "text": (
                "Mitochondria are often called the powerhouse of the cell because they generate "
                "most of the cell's ATP."
            ),
        },
        {
            "topic": "History",
            "text": (
                "The Renaissance marked the transition from the Middle Ages to modernity, "
                "spanning the 15th and 16th centuries in Europe."
            ),
        },
    ]

    logger.info("Inserting %d seed documents…", len(knowledge_base))
    inserted = []
    for doc in knowledge_base:
        res = await repo.insert_document(collection, doc)
        if not res.success or not res.data or "inserted_id" not in res.data:
            raise RuntimeError(f"Insert failed: {res.error}")
        inserted.append({"_id": res.data["inserted_id"], **doc})
    return inserted


async def embed_documents(embedder: ZMongoEmbedder, docs: List[Dict[str, Any]], embedding_field: str = "embeddings"):
    """
    Generates and stores chunked embeddings for each document's `text` field.
    """
    for d in docs:
        doc_id = d["_id"]
        text = d["text"]
        logger.info("Embedding and storing for _id=%s…", doc_id)
        res = await embedder.embed_and_store(doc_id, text, embedding_field=embedding_field)
        if not res.success:
            raise RuntimeError(f"embed_and_store failed: {res.error}")


async def run_query(retriever: ZMongoRetriever, query: str) -> List[Document]:
    """
    Executes the retriever via LangChain’s async interface.
    """
    logger.info("Running query: %s", query)
    docs = await retriever.ainvoke(query)
    return docs


async def main():
    # --- Configuration ---
    collection_name = "retriever_demo_knowledge_base"
    embedding_field = "embeddings"
    content_field = "text"
    query = "What is the fifth planet from the sun?"

    # Sanity check for keys (ZMongo itself will read MONGO_URI)
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit(
            "ERROR: GEMINI_API_KEY is not set. "
            "Put it in your environment or ~/resources/.env_local"
        )

    # --- Construct core components ---
    repo = ZMongo()
    embedder = ZMongoEmbedder(collection=collection_name)

    vector_searcher = LocalVectorSearch(
        repository=repo,
        collection=collection_name,
        embedding_field=embedding_field,
        chunked_embeddings=True,
        exact_rescore=True,     # use per-doc max-over-chunks rescoring
        use_hnsw=False,         # set True if you have hnswlib installed and want acceleration
        re_rank_candidates=None # let it compute candidates based on top_k
    )

    retriever = ZMongoRetriever(
        repository=repo,
        embedder=embedder,
        vector_searcher=vector_searcher,
        collection_name=collection_name,
        embedding_field=embedding_field,
        content_field=content_field,
        similarity_threshold=0.10,  # adjust for your data
        top_k=3
    )

    # --- Prepare data & embeddings ---
    inserted_docs = await prepare_data(repo, collection_name)
    await embed_documents(embedder, inserted_docs, embedding_field=embedding_field)

    # --- Query via LangChain retriever ---
    results = await run_query(retriever, query)

    # --- Display results ---
    print("\n=== Retrieval Results ===")
    if not results:
        print("No documents met the similarity threshold.")
    else:
        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}")
            print("-" * 40)
            print("Content:")
            print(doc.page_content)
            print("\nMetadata:")
            for k, v in doc.metadata.items():
                print(f"  {k}: {v}")

    # --- Optional: cleanup demo data (comment out to keep) ---
    # await repo.db.drop_collection(collection_name)
    repo.close()


if __name__ == "__main__":
    asyncio.run(main())
