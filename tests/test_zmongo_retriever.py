import os
import asyncio
from pathlib import Path
from typing import List

import pytest
import pytest_asyncio
import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document

# Adjust these imports to match your project's final structure
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_retriever.zmongo_toolbag.unified_vector_search import LocalVectorSearch
from zmongo_retriever.zmongo_toolbag.zmongo_retriever import ZMongoRetriever

# --- Test Configuration ---
load_dotenv(Path.home() / "resources" / ".env_local")

TEST_DB_NAME = "zmongo_retriever_test_db"
COLLECTION_NAME = "retriever_test_coll"
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

pytestmark = pytest.mark.skipif(
    not all([MONGO_URI, GEMINI_API_KEY]),
    reason="MONGO_URI and GEMINI_API_KEY must be set for live integration tests"
)


# --- Fixtures for Live Database Interaction ---

@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Creates a session-scoped event loop for all tests to share."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def motor_client(event_loop):
    """Provides a single Motor client for the entire test session."""
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    yield client
    client.close()


@pytest_asyncio.fixture
async def repository_instance(motor_client):
    """Provides a live ZMongo instance with a clean test database."""
    repo = ZMongo()
    yield repo
    repo.close()


@pytest_asyncio.fixture
def embedder_instance(repository_instance: ZMongo):
    """Provides a live ZMongoEmbedder instance."""
    return ZMongoEmbedder(
        collection=COLLECTION_NAME,
        gemini_api_key=GEMINI_API_KEY
    )


@pytest_asyncio.fixture
def vector_searcher_instance(repository_instance: ZMongo):
    """Provides a LocalVectorSearch instance for the retriever."""
    return LocalVectorSearch(
        repository=repository_instance,
        collection=COLLECTION_NAME,
        embedding_field="embeddings",
        chunked_embeddings=True,
        exact_rescore=True
    )


@pytest_asyncio.fixture
async def retriever_instance(repository_instance: ZMongo, embedder_instance: ZMongoEmbedder,
                             vector_searcher_instance: LocalVectorSearch):
    """Provides a fully configured ZMongoRetriever instance."""
    return ZMongoRetriever(
        repository=repository_instance,
        embedder=embedder_instance,
        vector_searcher=vector_searcher_instance,
        collection_name=COLLECTION_NAME,
        similarity_threshold=0.8,
        top_k=5
    )


# --- Helper Function ---

async def populate_test_data(repo: ZMongo, embedder: ZMongoEmbedder, documents: List[dict]):
    """Helper to insert and embed test documents."""
    texts_to_embed = [doc.get("text") for doc in documents if doc.get("text")]
    if texts_to_embed:
        embedding_results = await embedder.embed_texts_batched(texts_to_embed)
        for doc in documents:
            if doc.get("text") in embedding_results:
                doc["embeddings"] = embedding_results[doc["text"]]

    await repo.insert_documents(COLLECTION_NAME, documents)
    await asyncio.sleep(1)


# --- Test Cases ---

@pytest.mark.asyncio
async def test_retriever_initialization(retriever_instance: ZMongoRetriever):
    """Tests that the retriever initializes correctly with its dependencies."""
    assert isinstance(retriever_instance.repository, ZMongo)
    assert isinstance(retriever_instance.embedder, ZMongoEmbedder)
    assert isinstance(retriever_instance.vector_searcher, LocalVectorSearch)
    assert retriever_instance.collection_name == COLLECTION_NAME


@pytest.mark.asyncio
async def test_retrieval_flow_with_filtering(retriever_instance: ZMongoRetriever, repository_instance,
                                             embedder_instance):
    """
    Tests the primary retrieval path, ensuring results are correctly filtered.
    """
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
async def test_document_formatting_and_metadata(retriever_instance: ZMongoRetriever, repository_instance,
                                                embedder_instance):
    """
    Tests that retrieved documents are correctly formatted into LangChain
    Documents with the right page_content and metadata.
    """
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
async def test_no_results_found(retriever_instance: ZMongoRetriever):
    """Tests the scenario where no relevant documents are found."""
    results = await retriever_instance.ainvoke("Query with no possible results")
    assert results == []
