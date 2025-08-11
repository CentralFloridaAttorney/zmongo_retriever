import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from typing import AsyncGenerator

from bson.objectid import ObjectId
from langchain.schema import Document

from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zmongo_retriever import ZMongoRetriever
from dotenv import load_dotenv

load_dotenv(Path.home() / "resources" / ".env_local")

TEST_DB_NAME = "zmongo_retriever_test_db"
os.environ["MONGO_DATABASE_NAME"] = TEST_DB_NAME

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
requires_api_key = pytest.mark.skipif(
    not GEMINI_API_KEY, reason="This test requires a valid GEMINI_API_KEY environment variable."
)

IS_ATLAS_ENV = os.getenv("MONGO_IS_ATLAS", "false").lower() == "true"

@pytest_asyncio.fixture(scope="function")
async def zma() -> AsyncGenerator[ZMongoAtlas, None]:
    client = ZMongoAtlas()
    async with client as atlas_client:
        yield atlas_client
        await atlas_client.db.client.drop_database(TEST_DB_NAME)

@pytest.fixture(scope="function")
def coll_name() -> str:
    return f"retriever_test_{uuid.uuid4().hex}"

@pytest.fixture
def embedder(zma: ZMongoAtlas, coll_name: str) -> ZMongoEmbedder:
    return ZMongoEmbedder(collection=coll_name, repository=zma)

@pytest.fixture
def retriever(zma: ZMongoAtlas, embedder: ZMongoEmbedder, coll_name: str) -> ZMongoRetriever:
    return ZMongoRetriever(
        repository=zma,
        embedder=embedder,
        collection_name=coll_name,
        top_k=3
    )

async def seed_database(zma: ZMongoAtlas, embedder: ZMongoEmbedder, collection_name: str):
    await zma.create_vector_search_index(
        collection_name=collection_name,
        index_name="vector_index",
        embedding_field="embeddings",
        num_dimensions=768
    )

    docs_to_prepare = [
        {"text": "Apples are red.", "category": "fruit"},
        {"text": "Oranges are orange.", "category": "fruit"},
        {"text": "Bananas are yellow.", "category": "fruit"},
    ]

    docs_to_insert = []
    for doc in docs_to_prepare:
        embeddings_list = await embedder.embed_text(doc['text'])
        doc['embeddings'] = embeddings_list
        doc['_id'] = ObjectId()
        docs_to_insert.append(doc)

    await zma.insert_documents(collection_name, docs_to_insert)
    # FIX: Return the list of prepared documents, not the SafeResult
    return docs_to_insert

@requires_api_key
@pytest.mark.asyncio
async def test_retriever_finds_most_similar_document(retriever: ZMongoRetriever, zma: ZMongoAtlas,
                                                     embedder: ZMongoEmbedder, coll_name: str):
    await seed_database(zma, embedder, coll_name)
    query = "What color are apples?"
    results = await retriever.ainvoke(query)
    assert len(results) >= 1
    assert results[0].page_content == "Apples are red."
    assert results[0].metadata['retrieval_score'] > 0.8

@requires_api_key
@pytest.mark.asyncio
async def test_retriever_returns_top_k_documents(retriever: ZMongoRetriever, zma: ZMongoAtlas, embedder: ZMongoEmbedder,
                                                 coll_name: str):
    await seed_database(zma, embedder, coll_name)
    query = "Tell me about fruit salad."
    retriever.top_k = 2

    if not IS_ATLAS_ENV:
        retriever.similarity_threshold = 0.75

    results = await retriever.ainvoke(query)
    assert len(results) == 2
    result_contents = {doc.page_content for doc in results}
    assert "Apples are red." in result_contents
    assert "Oranges are orange." in result_contents

@requires_api_key
@pytest.mark.asyncio
async def test_retriever_returns_empty_list_for_no_matches(retriever: ZMongoRetriever, zma: ZMongoAtlas,
                                                           embedder: ZMongoEmbedder, coll_name: str):
    await seed_database(zma, embedder, coll_name)
    query = "The thunder struck the ground so hard that the earth shook."
    results = await retriever.ainvoke(query)
    assert len(results) == 0

@requires_api_key
@pytest.mark.asyncio
async def test_retriever_handles_empty_database(retriever: ZMongoRetriever):
    query = "Any documents here?"
    results = await retriever.ainvoke(query)
    assert results == []

@requires_api_key
@pytest.mark.asyncio
async def test_retrieved_document_metadata_format(retriever: ZMongoRetriever, zma: ZMongoAtlas,
                                                  embedder: ZMongoEmbedder,
                                                  coll_name: str):
    seeded_docs = await seed_database(zma, embedder, coll_name)
    apple_doc_id = str(seeded_docs[0]['_id'])
    query = "apple"
    results = await retriever.ainvoke(query)
    assert len(results) >= 1
    metadata = results[0].metadata
    assert metadata['source_document_id'] == apple_doc_id
    assert 'embeddings' not in metadata