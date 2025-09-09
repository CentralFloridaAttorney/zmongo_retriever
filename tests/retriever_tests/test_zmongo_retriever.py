import os
import asyncio
import logging
from pathlib import Path
from typing import List

import pytest
import pytest_asyncio
import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document

from zmongo_toolbag import ZRetriever
from zmongo_toolbag import ZMongo
from zmongo_toolbag import ZEmbedder
from zmongo_toolbag import LocalVectorSearch

# --- Test Configuration ---
load_dotenv(Path.home() / ".resources" / ".env_zmongo_retriever")
load_dotenv(Path.home() / ".resources" / ".secrets")

TEST_DB_NAME = "test"
COLLECTION_NAME = "test"
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

pytestmark = pytest.mark.skipif(
    not all([MONGO_URI, GEMINI_API_KEY]),
    reason="MONGO_URI and GEMINI_API_KEY must be set for live integration tests"
)


# --- Fixtures for Live Database Interaction ---

@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def motor_client(event_loop):
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    yield client
    client.close()


@pytest_asyncio.fixture
async def repository_instance(motor_client):
    """Provides a live ZMongo instance with a clean collection for each test."""
    repo = ZMongo()
    # FIX: Ensure the collection is dropped before each test for isolation
    await repo.db.drop_collection(COLLECTION_NAME)
    yield repo
    repo.close()


@pytest_asyncio.fixture
def embedder_instance(repository_instance: ZMongo):
    return ZEmbedder(
        repository=repository_instance,
        gemini_api_key=GEMINI_API_KEY
    )


@pytest_asyncio.fixture
def vector_searcher_instance(repository_instance: ZMongo):
    return LocalVectorSearch(
        repository=repository_instance,
        collection=COLLECTION_NAME,
        embedding_field="embeddings",
        chunked_embeddings=True,
        exact_rescore=True
    )


@pytest_asyncio.fixture
async def retriever_instance(repository_instance: ZMongo, embedder_instance: ZEmbedder,
                             vector_searcher_instance: LocalVectorSearch):
    return ZRetriever(
        repository=repository_instance,
        embedder=embedder_instance,
        vector_searcher=vector_searcher_instance,
        collection_name=COLLECTION_NAME,
        similarity_threshold=0.8,
        top_k=5,
        embedding_field='text',
    )


# --- Helper Function ---

async def populate_test_data(repo: ZMongo, embedder: ZEmbedder, documents: List[dict]):
    """
    FIX: Correctly inserts documents first, then calls the embedder to add
    embeddings to the now-existing documents in the database.
    """
    # 1. Insert the documents first.
    insert_res = await repo.insert_documents(COLLECTION_NAME, documents)
    if not insert_res.success:
        raise RuntimeError(f"Failed to insert documents: {insert_res.error}")

    # 2. Now, generate and store embeddings for each document.
    for doc in documents:
        doc_id = doc.get("_id")
        text_to_embed = doc.get("text")
        if doc_id and text_to_embed:
            embed_res = await embedder.embed_and_store(
                collection=COLLECTION_NAME,
                document_id=doc_id,
                text=text_to_embed,
                embedding_field="embeddings",
            )
            if not embed_res.success:
                logging.warning(f"Failed to embed document {doc_id}: {embed_res.error}")

    # Allow time for the vector searcher's index to refresh
    await asyncio.sleep(1)


# --- Test Cases ---

@pytest.mark.asyncio
async def test_retriever_initialization(retriever_instance: ZRetriever):
    assert isinstance(retriever_instance.repository, ZMongo)
    assert isinstance(retriever_instance.embedder, ZEmbedder)
    assert isinstance(retriever_instance.vector_searcher, LocalVectorSearch)
    assert retriever_instance.collection_name == COLLECTION_NAME


@pytest.mark.asyncio
async def test_retrieval_flow_with_filtering(retriever_instance: ZRetriever, repository_instance,
                                             embedder_instance):
    test_docs = [
        {"_id": ObjectId(), "text": "Python is a versatile programming language."},
        {"_id": ObjectId(), "text": "The sky is blue and the grass is green."},
        {"_id": ObjectId(), "text": "A dynamic, high-level, object-oriented language is Python."},
    ]
    await populate_test_data(repository_instance, embedder_instance, test_docs)

    query = "What is a good programming language?"
    results = await retriever_instance.ainvoke(query)

    assert len(results) == 2
    assert isinstance(results[0], Document)

    result_contents = {doc.page_content for doc in results}
    assert "Python is a versatile programming language." in result_contents
    assert "The sky is blue and the grass is green." not in result_contents


@pytest.mark.asyncio
async def test_document_formatting_and_metadata(retriever_instance: ZRetriever, repository_instance,
                                                embedder_instance):
    doc_id = ObjectId()
    test_doc = {
        "_id": doc_id,
        "text": "This is the main content.",
        "author": "Test Author",
        "category": "Testing"
    }
    await populate_test_data(repository_instance, embedder_instance, [test_doc])

    results = await retriever_instance.ainvoke("A query for the main content")

    assert len(results) == 1
    doc = results[0]

    assert doc.page_content == "This is the main content."
    assert doc.metadata["retrieval_score"] >= retriever_instance.similarity_threshold
    assert doc.metadata["_id"] == str(doc_id)
    assert doc.metadata["author"] == "Test Author"
    assert "embeddings" not in doc.metadata


@pytest.mark.asyncio
async def test_no_results_found(retriever_instance: ZRetriever):
    results = await retriever_instance.ainvoke("Query with no possible results")
    assert results == []
