import asyncio
import pytest
import numpy as np
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever import SafeResult
from zmongo_toolbag.zembedder import (
    ZEmbedder,
    EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    EMBEDDING_STYLE_RETRIEVAL_QUERY,
)

TEST_COLLECTION = "real_zembedder_integration"
TEST_FIELD = "embeddings"
TEST_TEXT = (
    "Florida foreclosure law requires strict compliance with statutory notice provisions. "
    "The borrower must be given an opportunity to cure the default before acceleration."
)

@pytest.fixture(scope="module")
def zmongo_instance():
    """Provide a real ZMongo instance connected to live database."""
    zm = ZMongo()
    zm.delete_all_documents(TEST_COLLECTION)
    yield zm
    zm.delete_all_documents(TEST_COLLECTION)
    zm.close()

@pytest.fixture(scope="module")
def zembedder_instance(zmongo_instance):
    """Initialize ZEmbedder with real model path (from ENV)."""
    embedder = ZEmbedder(repository=zmongo_instance)
    yield embedder
    embedder.close()

@pytest.mark.asyncio
async def test_query_embedding_generation(zembedder_instance):
    """Verify a single query text produces an embedding vector."""
    result = await zembedder_instance.get_embedding(
        text="What is the Florida notice requirement before foreclosure?",
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY,
        as_safe_result=True,
    )

    assert isinstance(result, SafeResult)
    assert result.success, result.error
    data = result.data
    assert "vectors" in data and len(data["vectors"]) > 0
    vec = np.array(data["vectors"][0])
    assert np.isfinite(vec).all()
    assert data["dimensionality"] == len(vec)


@pytest.mark.asyncio
async def test_document_embedding_and_cache(zmongo_instance, zembedder_instance):
    """Integration: full document embedding and cache reuse."""
    doc_id = ObjectId()
    zmongo_instance.insert_one(TEST_COLLECTION, {"_id": doc_id, "text": TEST_TEXT})

    # --- 1. Embed document and save to DB ---
    result = await zembedder_instance.get_embedding(
        text=TEST_TEXT,  # ✅ provide text explicitly
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection=TEST_COLLECTION,
        document_id=doc_id,
        embedding_field=TEST_FIELD,
        text_field="text",
        skip_if_present=False,
        as_safe_result=True,
    )
    assert result.success, f"Embedding failed: {result.error}"
    payload = result.data
    assert isinstance(payload["vectors"], list) and len(payload["vectors"]) > 0

    # --- 2. Confirm DB now contains embeddings ---
    find_res = zmongo_instance.find_document(TEST_COLLECTION, {"_id": doc_id})
    assert find_res.success and TEST_FIELD in find_res.data
    stored_vecs = find_res.data[TEST_FIELD]
    assert isinstance(stored_vecs, list) and len(stored_vecs) > 0

    # --- 3. Re-run with skip_if_present=True (should load from cache) ---
    cached_result = await zembedder_instance.get_embedding(
        text=TEST_TEXT,
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection=TEST_COLLECTION,
        document_id=doc_id,
        embedding_field=TEST_FIELD,
        text_field="text",
        skip_if_present=True,
        as_safe_result=True,
    )
    assert cached_result.success
    assert cached_result.data["from_cache"] is True
