import os
import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document
from pymongo.errors import OperationFailure

# Adjust these imports to match your project structure
# Assuming zmongo_atlas and data_processing exist, even if not provided
from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zmongo_retriever import ZMongoRetriever
from zmongo_toolbag.safe_result import SafeResult

# --- Test Configuration ---
TEST_DB_NAME = "zmongo_retriever_test_db"
COLLECTION_NAME = "retriever_test_coll"


# --- Fixtures ---

@pytest_asyncio.fixture
def mock_atlas_repo():
    """Mocks the ZMongoAtlas repository."""
    # spec=ZMongoAtlas ensures the mock has the same methods as the real class
    repo = AsyncMock(spec=ZMongoAtlas)
    return repo


@pytest_asyncio.fixture
def mock_embedder():
    """Mocks the ZMongoEmbedder."""
    embedder = AsyncMock(spec=ZMongoEmbedder)
    # Configure the mock to return a predictable embedding for any text
    embedder.embed_text.return_value = [[0.1, 0.2, 0.3, 0.4]]
    return embedder


@pytest_asyncio.fixture
def retriever_instance(mock_atlas_repo, mock_embedder):
    """Provides a ZMongoRetriever instance with mocked dependencies for each test."""
    retriever = ZMongoRetriever(
        repository=mock_atlas_repo,
        embedder=mock_embedder,
        collection_name=COLLECTION_NAME,
        similarity_threshold=0.8,  # Use a high threshold for precise testing
        top_k=5
    )
    return retriever


# --- Test Cases ---

@pytest.mark.asyncio
async def test_retriever_initialization(retriever_instance: ZMongoRetriever, mock_atlas_repo, mock_embedder):
    """Tests that the retriever initializes with the correct attributes."""
    assert retriever_instance.repository == mock_atlas_repo
    assert retriever_instance.embedder == mock_embedder
    assert retriever_instance.collection_name == COLLECTION_NAME
    assert retriever_instance.similarity_threshold == 0.8


@pytest.mark.asyncio
async def test_atlas_search_flow_with_filtering(retriever_instance: ZMongoRetriever, mock_atlas_repo: AsyncMock):
    """
    Tests the primary retrieval path using a mocked Atlas vector search,
    ensuring results are correctly filtered by the similarity threshold.
    """
    # Mock the repository to return documents with varying scores
    mock_search_results = [
        {"retrieval_score": 0.95, "document": {"_id": ObjectId(), "text": "High-scoring document."}},
        {"retrieval_score": 0.75, "document": {"_id": ObjectId(), "text": "Low-scoring document."}},
        {"retrieval_score": 0.85, "document": {"_id": ObjectId(), "text": "Medium-scoring document."}},
    ]
    mock_atlas_repo.vector_search.return_value = SafeResult.ok(mock_search_results)

    # Invoke the retriever
    query = "Find relevant documents"
    results = await retriever_instance.ainvoke(query)

    # Assertions
    mock_atlas_repo.vector_search.assert_called_once()
    assert len(results) == 2  # Only the docs with scores >= 0.8 should be returned
    assert isinstance(results[0], Document)
    assert results[0].page_content == "High-scoring document."
    assert results[0].metadata["retrieval_score"] == 0.95
    assert results[1].page_content == "Medium-scoring document."


@pytest.mark.asyncio
async def test_manual_search_fallback(retriever_instance: ZMongoRetriever, mock_atlas_repo: AsyncMock):
    """
    Tests that the retriever correctly falls back to manual search when
    Atlas Vector Search is not enabled (OperationFailure with code 31082).
    """
    # 1. Configure the Atlas search mock to fail with a specific error
    mock_atlas_repo.vector_search.side_effect = OperationFailure("SearchNotEnabled", code=31082)

    # 2. Configure the manual search mock (find_documents) to return results
    manual_search_results = [
        {"_id": ObjectId(), "text": "Manual result 1", "embeddings": [[0.1, 0.2, 0.3, 0.4]]},
        # Corrected: Use a truly dissimilar vector to ensure it fails the similarity check.
        {"_id": ObjectId(), "text": "Manual result 2", "embeddings": [[-0.9, -0.8, -0.7, -0.6]]},
    ]
    mock_atlas_repo.find_documents.return_value = SafeResult.ok(manual_search_results)

    # 3. Invoke the retriever
    query = "Trigger fallback"
    results = await retriever_instance.ainvoke(query)

    # 4. Assertions
    mock_atlas_repo.vector_search.assert_called_once()
    mock_atlas_repo.find_documents.assert_called_once()
    assert len(results) == 1
    assert results[0].page_content == "Manual result 1"
    # The score should be very close to 1.0 due to identical vectors
    assert results[0].metadata["retrieval_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_document_formatting_and_metadata(retriever_instance: ZMongoRetriever, mock_atlas_repo: AsyncMock):
    """
    Tests that the retrieved documents are correctly formatted into LangChain
    Documents with the right page_content and metadata.
    """
    doc_id = ObjectId()
    mock_result = {
        "retrieval_score": 0.9,
        "document": {
            "_id": doc_id,
            "text": "This is the main content.",
            "author": "Test Author",
            "category": "Testing",
            "embeddings": [[0.1, 0.2, 0.3, 0.4]] # This field should be excluded from metadata
        }
    }
    mock_atlas_repo.vector_search.return_value = SafeResult.ok([mock_result])

    results = await retriever_instance.ainvoke("A test query")

    assert len(results) == 1
    doc = results[0]

    assert doc.page_content == "This is the main content."
    assert doc.metadata["retrieval_score"] == 0.9
    assert doc.metadata["source_document_id"] == str(doc_id)
    assert doc.metadata["author"] == "Test Author"
    assert doc.metadata["category"] == "Testing"
    assert "embeddings" not in doc.metadata
    assert "text" not in doc.metadata


@pytest.mark.asyncio
async def test_no_results_found(retriever_instance: ZMongoRetriever, mock_atlas_repo: AsyncMock):
    """Tests the scenario where no relevant documents are found."""
    mock_atlas_repo.vector_search.return_value = SafeResult.ok([])
    results = await retriever_instance.ainvoke("Query with no results")
    assert results == []


@pytest.mark.asyncio
async def test_no_embedding_for_query(retriever_instance: ZMongoRetriever, mock_embedder: AsyncMock):
    """Tests the case where the embedder fails to generate an embedding."""
    mock_embedder.embed_text.return_value = [] # Simulate embedding failure
    results = await retriever_instance.ainvoke("This query will not be embedded")
    assert results == []
    retriever_instance.repository.vector_search.assert_not_called()
