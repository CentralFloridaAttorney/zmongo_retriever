import asyncio
import os
from pathlib import Path
from typing import List
from unittest.mock import patch, AsyncMock

import pytest
import pytest_asyncio
import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document

# Adjust these imports to match your project structure
from zmongo_atlas import ZMongoAtlas
from zmongo_embedder import ZMongoEmbedder
from zmongo import ZMongo
from zmongo_embedding_processor import ZMongoProcessor

# --- Test Configuration ---
# Load environment variables from a .env file for local development
load_dotenv(Path.home() / "resources" / ".env_local")

TEST_DB_NAME = "zmongo_processor_test_db"
COLLECTION_NAME = "processor_test_coll"
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Skip all tests in this file if the necessary environment variables are not set
pytestmark = pytest.mark.skipif(
    not all([MONGO_URI, GEMINI_API_KEY]),
    reason="MONGO_URI and GEMINI_API_KEY must be set in the environment for live tests"
)


# --- Fixtures ---

@pytest_asyncio.fixture
async def zmongo_atlas_instance():
    """Provides a live ZMongoAtlas instance connected to a clean test database for each test."""
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    test_db = client[TEST_DB_NAME]
    repo = ZMongoAtlas(db=test_db)
    yield repo
    # Teardown: Drop the entire test database after each test
    await client.drop_database(TEST_DB_NAME)
    await repo.close()


@pytest_asyncio.fixture
def processor_instance(zmongo_atlas_instance: ZMongoAtlas):
    """Provides a ZMongoProcessor instance for each test."""
    return ZMongoProcessor(
        collection_name=COLLECTION_NAME,
        text_field_keys=["title", "content", "metadata.author"],
        mongo_atlas=zmongo_atlas_instance,
        gemini_api_key=GEMINI_API_KEY
    )


# --- Helper Function ---

async def populate_test_data(repo: ZMongoAtlas, documents: List[dict]):
    """Helper to insert test documents into the collection."""
    await repo.insert_documents(COLLECTION_NAME, documents)


# --- Test Cases ---

@pytest.mark.asyncio
async def test_processor_initialization(processor_instance: ZMongoProcessor):
    """Tests that the processor initializes correctly."""
    assert isinstance(processor_instance.repository, ZMongoAtlas)
    assert processor_instance.collection_name == COLLECTION_NAME
    assert processor_instance.text_field_keys == ["title", "content", "metadata.author"]


@pytest.mark.asyncio
async def test_get_embedding_field_name(processor_instance: ZMongoProcessor):
    """Tests the dynamic creation of embedding field names."""
    assert processor_instance._get_embedding_field_name("title") == "title_embedding"
    assert processor_instance._get_embedding_field_name("metadata.author") == "metadata_author_embedding"


@pytest.mark.asyncio
async def test_process_and_embed_collection(processor_instance: ZMongoProcessor, zmongo_atlas_instance: ZMongoAtlas):
    """
    Tests the core embedding process, ensuring it finds documents and calls the embedder.
    """
    # 1. Setup: Populate with documents that need processing
    test_docs = [
        {"_id": ObjectId(), "title": "First Doc", "content": "Some text here."},
        {"_id": ObjectId(), "title": "Second Doc", "content": "More text.", "metadata": {"author": "Test Author"}},
        {"_id": ObjectId(), "title": "Third Doc", "content": "Already done.", "title_embedding": [[0.1]]} # Should be skipped for 'title'
    ]
    await populate_test_data(zmongo_atlas_instance, test_docs)

    # 2. Mock the embedder's store method to track calls
    with patch.object(processor_instance.embedder, 'embed_and_store', new_callable=AsyncMock) as mock_embed_store:
        # 3. Action: Run the processing
        summary = await processor_instance.process_and_embed_collection()

        # 4. Assertions
        # Corrected counts:
        # title: finds 2 docs to process
        # content: finds 3 docs to process
        # metadata.author: finds 3 docs to process
        # Total checked = 2 + 3 + 3 = 8
        assert summary["total_docs_checked"] == 8
        # Embeddings created = 2 (title) + 3 (content) + 1 (author, since only one doc has it) = 6
        assert summary["embeddings_created"] == 6
        assert summary["total_failures"] == 0
        assert mock_embed_store.call_count == 6


@pytest.mark.asyncio
async def test_process_with_limit(processor_instance: ZMongoProcessor, zmongo_atlas_instance: ZMongoAtlas):
    """Tests that the 'limit' parameter is respected during processing."""
    # 1. Setup: Populate with more documents than the limit
    test_docs = [
        {"_id": ObjectId(), "content": f"Content {i}"} for i in range(5)
    ]
    await populate_test_data(zmongo_atlas_instance, test_docs)

    # 2. Mock and Action
    with patch.object(processor_instance.embedder, 'embed_and_store', new_callable=AsyncMock) as mock_embed_store:
        # Process only 2 documents for the 'content' field
        await processor_instance.process_and_embed_collection(limit=2)

        # 3. Assertions
        # It should only call the embedder for the 2 documents it found
        assert mock_embed_store.call_count == 2


@pytest.mark.asyncio
async def test_search_functionality(processor_instance: ZMongoProcessor):
    """Tests that the search method correctly initializes and calls the retriever."""
    query = "test query"
    search_field = "content"
    mock_retriever_path = "zmongo_embedding_processor.ZMongoRetriever"

    # 1. Mock the ZMongoRetriever class
    with patch(mock_retriever_path) as MockRetriever:
        # Configure the mock instance that will be created
        mock_instance = MockRetriever.return_value
        # FIX: Use an AsyncMock for the ainvoke method to make it awaitable
        mock_instance.ainvoke = AsyncMock(return_value=[Document(page_content="mocked result")])

        # 2. Action: Call the search method
        results = await processor_instance.search(query, search_field)

        # 3. Assertions
        # Check that the retriever was initialized with the correct parameters
        MockRetriever.assert_called_once_with(
            repository=processor_instance.repository,
            embedder=processor_instance.embedder,
            collection_name=COLLECTION_NAME,
            embedding_field="content_embedding",
            content_field="content",
            top_k=5,
            similarity_threshold=0.7
        )

        # Check that the retriever's invoke method was called with the query
        mock_instance.ainvoke.assert_called_once_with(query)

        # Check that the results are passed through correctly
        assert len(results) == 1
        assert results[0].page_content == "mocked result"


@pytest.mark.asyncio
async def test_search_with_invalid_field(processor_instance: ZMongoProcessor):
    """Tests that searching with a non-configured field raises a ValueError."""
    with pytest.raises(ValueError, match="'invalid_field' is not one of the configured text fields"):
        await processor_instance.search("test query", "invalid_field")
