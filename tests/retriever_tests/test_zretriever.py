import os
import asyncio
import logging
from pathlib import Path
from typing import List

import pytest
import pytest_asyncio
from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document

# Import the classes to be tested
from zmongo_toolbag.zretriever import ZRetriever
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import ZEmbedder, EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
from zmongo_toolbag.unified_vector_search import LocalVectorSearch

# --- Test Configuration ---
load_dotenv(Path.home() / "resources" / ".env_local")

COLLECTION_NAME = "retriever_test_coll"
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Skip all tests in this file if the required environment variables are not set
pytestmark = pytest.mark.skipif(
    not all([MONGO_URI, GEMINI_API_KEY]),
    reason="MONGO_URI and GEMINI_API_KEY must be set for live integration tests"
)


# --- Fixtures for Live Database Interaction ---

@pytest_asyncio.fixture(scope="module")
def event_loop():
    """Creates a module-scoped event loop for all tests to share."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def repository_instance():
    """Provides a live ZMongo instance with a clean collection for each test."""
    repo = ZMongo()
    await repo.db.drop_collection(COLLECTION_NAME)
    yield repo
    repo.close()


@pytest_asyncio.fixture
def embedder_instance(repository_instance: ZMongo):
    """Provides a live ZEmbedder instance."""
    return ZEmbedder(
        repository=repository_instance,
        gemini_api_key=GEMINI_API_KEY
    )


@pytest_asyncio.fixture
def vector_searcher_instance(repository_instance: ZMongo):
    """Provides a live LocalVectorSearch instance."""
    return LocalVectorSearch(
        repository=repository_instance,
        collection=COLLECTION_NAME,
        embedding_field="embeddings",
    )


@pytest_asyncio.fixture
async def retriever_instance(repository_instance: ZMongo, embedder_instance: ZEmbedder,
                             vector_searcher_instance: LocalVectorSearch):
    """Provides a fully configured ZRetriever instance."""
    return ZRetriever(
        repository=repository_instance,
        embedder=embedder_instance,
        vector_searcher=vector_searcher_instance,
        collection_name=COLLECTION_NAME,
        embedding_field="embeddings",
        similarity_threshold=0.8,
        top_k=5,
        query_embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
    )


# --- Helper Function ---

async def populate_test_data(retriever: ZRetriever, documents: List[dict]):
    """
    Correctly inserts documents first, then calls the embedder to add
    embeddings to the now-existing documents in the database.
    """
    repo = retriever.repository
    embedder = retriever.embedder
    embedding_field = retriever.embedding_field

    insert_res = await repo.insert_documents(COLLECTION_NAME, documents)
    if not insert_res.success:
        raise RuntimeError(f"Failed to insert documents: {insert_res.error}")

    for doc in documents:
        doc_id = doc.get("_id")
        text_to_embed = doc.get("text")
        if doc_id and text_to_embed:
            embed_res = await embedder.embed_and_store(
                collection=COLLECTION_NAME,
                document_id=doc_id,
                text=text_to_embed,
                embedding_field=embedding_field,
            )
            if not embed_res.success:
                logging.warning(f"Failed to embed document {doc_id}: {embed_res.error}")

    # Allow time for the vector searcher's in-memory index to refresh
    await asyncio.sleep(1)


# --- Test Cases ---

@pytest.mark.asyncio
async def test_retriever_initialization(retriever_instance: ZRetriever):
    """Tests that the retriever and its components are initialized correctly."""
    assert isinstance(retriever_instance, ZRetriever)
    assert isinstance(retriever_instance.repository, ZMongo)
    assert isinstance(retriever_instance.embedder, ZEmbedder)
    assert isinstance(retriever_instance.vector_searcher, LocalVectorSearch)
    assert retriever_instance.collection_name == COLLECTION_NAME


@pytest.mark.asyncio
async def test_retrieval_flow_with_filtering(retriever_instance: ZRetriever):
    """
    Tests the primary retrieval path, ensuring results are correctly filtered
    by the similarity threshold.
    """
    test_docs = [
        {"_id": ObjectId(), "text": "Python is a versatile programming language."},
        {"_id": ObjectId(), "text": "The sky is blue and the grass is green."},
        {"_id": ObjectId(), "text": "A dynamic, high-level, object-oriented language is Python."},
    ]
    await populate_test_data(retriever_instance, test_docs)

    query = "What is a good programming language?"
    results = await retriever_instance.ainvoke(query)

    assert len(results) == 2
    assert isinstance(results[0], Document)

    result_contents = {doc.page_content for doc in results}
    assert "Python is a versatile programming language." in result_contents
    assert "The sky is blue and the grass is green." not in result_contents


@pytest.mark.asyncio
async def test_document_formatting_and_metadata(retriever_instance: ZRetriever):
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
    await populate_test_data(retriever_instance, [test_doc])

    results = await retriever_instance.ainvoke("A query for the main content")

    assert len(results) == 1
    doc = results[0]

    assert doc.page_content == "This is the main content."
    assert doc.metadata["retrieval_score"] >= retriever_instance.similarity_threshold
    assert doc.metadata["_id"] == str(doc_id)
    assert doc.metadata["author"] == "Test Author"
    assert "embeddings" not in doc.metadata, "Embedding field should be excluded from metadata"


@pytest.mark.asyncio
async def test_no_results_found(retriever_instance: ZRetriever):
    """Tests the scenario where no relevant documents are found."""
    # Populate with irrelevant data first
    await populate_test_data(retriever_instance, [{"_id": ObjectId(), "text": "Irrelevant fact."}])

    results = await retriever_instance.ainvoke("A query that will not match anything")
    assert results == []
