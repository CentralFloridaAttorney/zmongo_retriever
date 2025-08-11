import asyncio
import os
from pathlib import Path
from typing import List

import pytest
import pytest_asyncio
import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document

# Adjust these imports to match your project structure
from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zmongo_retriever import ZMongoRetriever
from safe_result import SafeResult

# --- Test Configuration ---
# Load environment variables from a .env file
load_dotenv(Path.home() / "resources" / ".env_local")

# Use a dedicated database for testing to avoid conflicts with real data
TEST_DB_NAME = "zmongo_retriever_test_db"
COLLECTION_NAME = "retriever_test_coll"
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Skip all tests in this file if the necessary environment variables are not set
pytestmark = pytest.mark.skipif(
    not all([MONGO_URI, GEMINI_API_KEY]),
    reason="MONGO_URI and GEMINI_API_KEY must be set in the environment for live tests"
)


# --- Fixtures for Live Database Interaction ---

@pytest_asyncio.fixture
async def zmongo_atlas_instance():
    """
    Provides a live ZMongoAtlas instance connected to the test database.
    This fixture now runs for each test to ensure a fresh connection and event loop.
    """
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    test_db = client[TEST_DB_NAME]

    repo = ZMongoAtlas(db=test_db)
    yield repo

    # Teardown: Drop the entire test database and close the connection
    await client.drop_database(TEST_DB_NAME)
    await repo.close()


@pytest_asyncio.fixture
def zmongo_embedder_instance(zmongo_atlas_instance: ZMongoAtlas):
    """Provides a live ZMongoEmbedder instance for each test."""
    return ZMongoEmbedder(
        repository=zmongo_atlas_instance,
        collection=COLLECTION_NAME,
        # Add the missing argument here
        page_content_key="text",
        gemini_api_key=GEMINI_API_KEY
    )


@pytest_asyncio.fixture
async def retriever_instance(zmongo_atlas_instance: ZMongoAtlas, zmongo_embedder_instance: ZMongoEmbedder):
    """
    Provides a ZMongoRetriever instance with live dependencies.
    The collection is implicitly cleaned up by dropping the database in the atlas fixture.
    """
    yield ZMongoRetriever(
        repository=zmongo_atlas_instance,
        embedder=zmongo_embedder_instance,
        collection_name=COLLECTION_NAME,
        similarity_threshold=0.8,  # Increased threshold to avoid false positives
        top_k=5
    )


# --- Helper Function ---

async def populate_test_data(repo: ZMongoAtlas, embedder: ZMongoEmbedder, documents: List[dict]):
    """Helper to insert and embed test documents."""
    for doc in documents:
        text_to_embed = doc.get("text")
        if text_to_embed:
            # Generate real embeddings
            embeddings = await embedder.embed_text(text_to_embed)
            doc["embeddings"] = embeddings
    await repo.insert_documents(COLLECTION_NAME, documents)


# --- Test Cases ---

@pytest.mark.asyncio
async def test_retriever_initialization(retriever_instance: ZMongoRetriever):
    """Tests that the retriever initializes correctly with live instances."""
    assert isinstance(retriever_instance.repository, ZMongoAtlas)
    assert isinstance(retriever_instance.embedder, ZMongoEmbedder)
    assert retriever_instance.collection_name == COLLECTION_NAME


@pytest.mark.asyncio
async def test_retrieval_flow_with_filtering(retriever_instance: ZMongoRetriever, zmongo_atlas_instance,
                                             zmongo_embedder_instance):
    """
    Tests the primary retrieval path, ensuring results are correctly filtered
    by the similarity threshold.
    """
    # 1. Setup: Populate the database with test data
    test_docs = [
        {"_id": ObjectId(), "text": "Python is a versatile programming language."},
        {"_id": ObjectId(), "text": "The sky is blue and the grass is green."},
        {"_id": ObjectId(), "text": "A dynamic, high-level, object-oriented language is Python."},
    ]
    await populate_test_data(zmongo_atlas_instance, zmongo_embedder_instance, test_docs)

    # 2. Action: Invoke the retriever with a relevant query
    query = "What is a good programming language?"
    results = await retriever_instance.ainvoke(query)

    # 3. Assertions
    assert len(results) == 2  # Should find the two Python-related documents
    assert isinstance(results[0], Document)

    # Check that the scores are above the threshold
    for doc in results:
        assert doc.metadata["retrieval_score"] >= retriever_instance.similarity_threshold

    # Check that the content is correct
    result_contents = {doc.page_content for doc in results}
    assert "Python is a versatile programming language." in result_contents
    assert "A dynamic, high-level, object-oriented language is Python." in result_contents
    assert "The sky is blue and the grass is green." not in result_contents


@pytest.mark.asyncio
async def test_document_formatting_and_metadata(retriever_instance: ZMongoRetriever, zmongo_atlas_instance,
                                                zmongo_embedder_instance):
    """
    Tests that retrieved documents are correctly formatted into LangChain
    Documents with the right page_content and metadata.
    """
    # 1. Setup
    doc_id = ObjectId()
    test_doc = {
        "_id": doc_id,
        "text": "This is the main content.",
        "author": "Test Author",
        "category": "Testing"
    }
    await populate_test_data(zmongo_atlas_instance, zmongo_embedder_instance, [test_doc])

    # 2. Action
    results = await retriever_instance.ainvoke("A query for the main content")

    # 3. Assertions
    assert len(results) == 1
    doc = results[0]

    assert doc.page_content == "This is the main content."
    assert doc.metadata["retrieval_score"] >= retriever_instance.similarity_threshold
    assert doc.metadata["_id"] == str(doc_id)  # Check for the original _id
    assert doc.metadata["author"] == "Test Author"
    assert doc.metadata["category"] == "Testing"
    assert "embeddings" not in doc.metadata  # Ensure embeddings are excluded


@pytest.mark.asyncio
async def test_no_results_found(retriever_instance: ZMongoRetriever):
    """Tests the scenario where no relevant documents are found."""
    # No data is populated for this test, so the collection is empty
    results = await retriever_instance.ainvoke("Query with no possible results")
    assert results == []
