import os
import uuid
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv

from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

load_dotenv(r"C:\Users\iriye\resources\.env_local")
TEST_DB_NAME = "zmongo_atlas_test_db"
os.environ["MONGO_DATABASE_NAME"] = TEST_DB_NAME

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
requires_api_key = pytest.mark.skipif(
    not GEMINI_API_KEY, reason="This test requires a valid GEMINI_API_KEY."
)

@pytest_asyncio.fixture(scope="function")
async def zma() -> AsyncGenerator[ZMongoAtlas, None]:
    client = ZMongoAtlas()
    async with client as atlas_client:
        yield atlas_client
        await atlas_client.db.client.drop_database(TEST_DB_NAME)

@pytest.fixture(scope="function")
def coll_name() -> str:
    return f"atlas_test_{uuid.uuid4().hex}"

@pytest.mark.asyncio
async def test_create_and_list_search_index(zma: ZMongoAtlas, coll_name: str):
    """
    Tests creating a vector search index. In a local environment, this
    should succeed gracefully by skipping the actual creation.
    """
    index_name = "test_vector_index"
    create_result = await zma.create_vector_search_index(
        collection_name=coll_name,
        index_name=index_name,
        embedding_field="test_embeddings",
        num_dimensions=768
    )
    # FIX: Correct assertion to check for success
    assert create_result.success, f"Index creation failed: {create_result.error}"

@requires_api_key
@pytest.mark.asyncio
async def test_vector_search(zma: ZMongoAtlas, coll_name: str):
    """
    Tests the end-to-end vector search functionality, which will use the
    manual fallback logic when run locally.
    """
    index_name = "search_test_index"
    embedding_field = "embeddings"

    # This will be skipped gracefully in a local environment
    await zma.create_vector_search_index(
        collection_name=coll_name, index_name=index_name,
        embedding_field=embedding_field, num_dimensions=768
    )

    embedder = ZMongoEmbedder(collection=coll_name, repository=zma)
    docs_to_insert = [
        {"text": "The sky is blue.", "category": "nature"},
        {"text": "An orange is a type of citrus fruit.", "category": "food"},
    ]
    for doc in docs_to_insert:
        embeddings = await embedder.embed_text(doc["text"])
        doc[embedding_field] = embeddings
        await zma.insert_document(coll_name, doc)

    query = "What color is the sky?"
    query_embedding = (await embedder.embed_text(query))[0]

    search_result = await zma.vector_search(
        collection_name=coll_name, query_vector=query_embedding,
        index_name=index_name, embedding_field=embedding_field, top_k=1
    )

    # FIX: Correct assertions to check for success and valid data
    assert search_result.success, f"Vector search failed: {search_result.error}"
    assert len(search_result.data) == 1

    top_result = search_result.data[0]
    assert "retrieval_score" in top_result
    assert top_result["retrieval_score"] > 0.8

    retrieved_doc = top_result.get("document", {})
    assert retrieved_doc.get("text") == "The sky is blue."