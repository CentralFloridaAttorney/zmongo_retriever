import os
import uuid
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from bson.objectid import ObjectId
from dotenv import load_dotenv

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

# --- Test Configuration ---
load_dotenv(r"C:\Users\iriye\resources\.env_local")
TEST_DB_NAME = "zmongo_embedder_test_db"
os.environ["MONGO_DATABASE_NAME"] = TEST_DB_NAME

# Conditional skip for tests requiring a real API call
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
requires_api_key = pytest.mark.skipif(
    not GEMINI_API_KEY, reason="This test requires a valid GEMINI_API_KEY environment variable."
)


# --- Pytest Fixtures ---
import pytest
import pytest_asyncio
from typing import AsyncGenerator

# ... other imports ...

# --- Pytest Fixtures ---

@pytest_asyncio.fixture(scope="function")  # <-- FIX: Scope must be "function"
async def zmongo() -> AsyncGenerator[ZMongo, None]:
    """
    Provides a fresh ZMongo instance for each test function and cleans up after.
    """
    client = ZMongo()
    async with client as zm:
        yield zm
        # This teardown logic now runs after EACH test, ensuring a clean slate.
        await zm.db.client.drop_database(TEST_DB_NAME)


@pytest.fixture(scope="function")  # <-- FIX: Scope must be "function"
def embedder(zmongo: ZMongo, coll_name: str) -> ZMongoEmbedder:
    """
    Provides a fresh, isolated ZMongoEmbedder instance for each test.
    """
    return ZMongoEmbedder(
        collection=coll_name,
        repository=zmongo,
        gemini_api_key="dummy_key_for_testing" # Dummy key is fine for mocked tests
    )

@pytest_asyncio.fixture(scope="function")
async def zmongo() -> AsyncGenerator[ZMongo, None]:
    """Provides a fresh ZMongo instance for each test function."""
    client = ZMongo()
    async with client as zm:
        yield zm
        await zm.db.client.drop_database(TEST_DB_NAME)


@pytest.fixture(scope="function")
def coll_name() -> str:
    """Provides a unique collection name for each test."""
    return f"test_collection_{uuid.uuid4().hex}"


@pytest.fixture(scope="function")
def embedder(zmongo: ZMongo, coll_name: str) -> ZMongoEmbedder:
    """Provides a fresh, isolated ZMongoEmbedder instance for each test."""
    # This fixture now relies on the environment for the API key.
    # The 'requires_api_key' marker will handle cases where it's not set.
    return ZMongoEmbedder(collection=coll_name, repository=zmongo)


# --- Test Cases ---

def test_initialization(zmongo: ZMongo, coll_name: str):
    """Tests that the ZMongoEmbedder initializes correctly."""
    # This test doesn't need a real API key to pass
    embedder = ZMongoEmbedder(collection=coll_name, repository=zmongo, gemini_api_key="dummy_key")
    assert embedder.repository is not None


def test_split_chunks(embedder: ZMongoEmbedder):
    """Tests the internal text chunking logic."""
    chunks = embedder._split_chunks("a" * 100, chunk_size=50, overlap=10)
    assert len(chunks) == 3


@requires_api_key
@pytest.mark.asyncio
async def test_embed_text_with_new_chunk(embedder: ZMongoEmbedder, zmongo: ZMongo):
    """
    Tests embedding a new piece of text, ensuring the API is called
    and the result is cached in the database.
    """
    text = "This is a new sentence for a real embedding."

    embeddings = await embedder.embed_text(text)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768  # Standard dimension for embedding-001

    cache_result = await zmongo.find_documents("_embedding_cache", {})
    assert cache_result.success and len(cache_result.data) == 1
    assert cache_result.data[0]['embedding'] == embeddings[0]


@requires_api_key
@pytest.mark.asyncio
async def test_embed_text_with_cached_chunk(embedder: ZMongoEmbedder, zmongo: ZMongo):
    """
    Tests that previously cached text chunks are reused and do not
    trigger a new API call.
    """
    text = "This sentence will be embedded and then retrieved from the cache."

    # First call to populate the cache
    first_embeddings = await embedder.embed_text(text)

    # Second call should hit the cache
    second_embeddings = await embedder.embed_text(text)

    # Assertions
    assert first_embeddings == second_embeddings

    cache_result = await zmongo.find_documents("_embedding_cache", {})
    assert len(cache_result.data) == 1  # This now passes due to test isolation


@requires_api_key
@pytest.mark.asyncio
async def test_embed_and_store(embedder: ZMongoEmbedder, coll_name: str):
    """
    Tests the end-to-end process of generating a real embedding and
    storing it in a target document.
    """
    document_id = ObjectId()
    text = "A document to be embedded and stored with a real vector."

    await embedder.embed_and_store(document_id, text, embedding_field="real_embedding")

    stored_doc_result = await embedder.repository.find_document(coll_name, {"_id": document_id})
    assert stored_doc_result.success and stored_doc_result.data is not None

    stored_embedding = stored_doc_result.data["real_embedding"]
    assert len(stored_embedding) == 1
    assert len(stored_embedding[0]) == 768