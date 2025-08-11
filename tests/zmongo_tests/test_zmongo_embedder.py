import os
import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv

# Adjust imports to match the application's structure to avoid TypeErrors
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

# --- Test Configuration ---
# Load environment variables from a .env file for local development
load_dotenv(Path.home() / "resources" / ".env_local")

TEST_DB_NAME = "zmongo_embedder_test_db"
COLLECTION_NAME = "test_embedding_coll"
CACHE_COLLECTION = "_embedding_cache"


# --- Fixtures ---

@pytest_asyncio.fixture(scope="function")
async def zmongo_instance():
    """Provides a clean ZMongo instance for each test."""
    os.environ["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
    os.environ["MONGO_DATABASE_NAME"] = TEST_DB_NAME
    zmongo = ZMongo()

    # Setup: clean database and cache
    collections = await zmongo.db.list_collection_names()
    for coll in collections:
        await zmongo.db.drop_collection(coll)

    # Use the dedicated clear_cache method for a robust cleanup
    if hasattr(zmongo, 'clear_cache'):
        await zmongo.clear_cache()

    yield zmongo

    # Teardown: clean database and cache
    collections = await zmongo.db.list_collection_names()
    for coll in collections:
        await zmongo.db.drop_collection(coll)

    if hasattr(zmongo, 'clear_cache'):
        await zmongo.clear_cache()


@pytest_asyncio.fixture
async def embedder(zmongo_instance):
    """Provides a ZMongoEmbedder instance with a dummy API key for initialization."""
    # Add the missing 'page_content_key' argument
    return ZMongoEmbedder(
        repository=zmongo_instance,
        collection=COLLECTION_NAME,
        page_content_key="text_content",  # Or whatever key you use
        gemini_api_key="DUMMY_KEY"
    )

# --- Test Cases for ZMongoEmbedder ---

@pytest.mark.asyncio
async def test_embedder_initialization(embedder: ZMongoEmbedder, zmongo_instance: ZMongo):
    """Tests that the embedder initializes correctly."""
    assert embedder.repository == zmongo_instance
    assert embedder.collection == COLLECTION_NAME
    assert embedder.embedding_model_name == "models/embedding-001"


@pytest.mark.asyncio
async def test_split_chunks(zmongo_instance: ZMongo):
    """Tests the internal text chunking logic."""
    embedder = ZMongoEmbedder(repository=zmongo_instance, collection="dummy", page_content_key='content', gemini_api_key="dummy")
    text = "a" * 2000
    chunks = embedder._split_chunks(text, chunk_size=1000, overlap=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    # Corrected assertion: 2000 total length, start of chunk 2 is 900, start of chunk 3 is 1800.
    # Remainder is 2000 - 1800 = 200.
    assert len(chunks[2]) == 200
    assert chunks[1].startswith("a" * 100)  # Checks overlap


@pytest.mark.asyncio
async def test_embed_text_and_caching(embedder: ZMongoEmbedder):
    """Tests that text is embedded and the result is cached."""
    text_to_embed = "This is a test sentence for embedding."

    # Mock the internal API call method for this specific test
    with patch.object(embedder, '_get_embedding_from_api', new_callable=AsyncMock) as mock_api_call:
        mock_api_call.return_value = [0.1, 0.2, 0.3]

        # --- First call: Should call the mocked method and cache the result ---
        embeddings = await embedder.embed_text(text_to_embed)

        # Assertions for the first call
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        mock_api_call.assert_called_once()  # Internal API method was called

        # Check that the embedding was cached in MongoDB
        cache_res = await embedder.repository.find_document(CACHE_COLLECTION, {})
        assert cache_res.success
        assert cache_res.data is not None
        assert cache_res.data["embedding"] == [0.1, 0.2, 0.3]

        # --- Second call: Should use the cache and NOT call the API method ---
        mock_api_call.reset_mock()  # Reset the call counter
        embeddings_from_cache = await embedder.embed_text(text_to_embed)

        # Assertions for the second call
        assert embeddings_from_cache == embeddings
        mock_api_call.assert_not_called()  # Internal API method was NOT called again


@pytest.mark.asyncio
async def test_embed_and_store(embedder: ZMongoEmbedder):
    """Tests embedding text and storing it in a target document."""
    # 1. Create a target document to store the embedding in
    insert_res = await embedder.repository.insert_document(COLLECTION_NAME, {"text_content": "some original text"})
    doc_id = insert_res.data.inserted_id

    # 2. Mock the internal API call and run the embed_and_store process
    with patch.object(embedder, '_get_embedding_from_api', new_callable=AsyncMock) as mock_api_call:
        mock_api_call.return_value = [0.1, 0.2, 0.3]
        text_to_embed = "This will be embedded and stored."
        await embedder.embed_and_store(document_id=doc_id, text=text_to_embed, embedding_field="my_embeddings")

    # 3. Verify the document was updated correctly
    updated_doc_res = await embedder.repository.find_document(COLLECTION_NAME, {"_id": doc_id})
    assert updated_doc_res.success
    updated_doc = updated_doc_res.data

    assert "my_embeddings" in updated_doc
    assert len(updated_doc["my_embeddings"]) == 1
    assert updated_doc["my_embeddings"][0] == [0.1, 0.2, 0.3]
    assert updated_doc["text_content"] == "some original text"  # Ensure other fields are preserved


@pytest.mark.asyncio
async def test_embed_and_store_with_string_id(embedder: ZMongoEmbedder):
    """Tests that embed_and_store works correctly when given a string ID."""
    insert_res = await embedder.repository.insert_document(COLLECTION_NAME, {"data": "test"})
    doc_id_obj = insert_res.data.inserted_id
    doc_id_str = str(doc_id_obj)

    with patch.object(embedder, '_get_embedding_from_api', new_callable=AsyncMock) as mock_api_call:
        mock_api_call.return_value = [0.1, 0.2, 0.3]
        await embedder.embed_and_store(document_id=doc_id_str, text="testing string id")

    updated_doc_res = await embedder.repository.find_document(COLLECTION_NAME, {"_id": doc_id_obj})
    assert updated_doc_res.success
    assert "embeddings" in updated_doc_res.data


@pytest.mark.asyncio
async def test_embed_and_store_with_invalid_id(embedder: ZMongoEmbedder, capsys):
    """Tests that a graceful message is printed for an invalid string ID."""
    await embedder.embed_and_store(document_id="not-a-valid-id", text="test")
    captured = capsys.readouterr()
    assert "is not a valid ObjectId" in captured.out
